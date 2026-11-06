# Tiled low-rank Schur contribution for ClusteredLowRankSolver 1.0.3 (symmetric low-rank blocks with one
# subblock). For A_p = sum_r lambda_pr v_pr v_pr^T:
#   <A_p, X^-1 A_q Y> = sum_{r1,r2} lambda_pr1 lambda_qr2 (u_pr1 . u_qr2)(w_pr1 . w_qr2),
#   u = L_X^-1 v,  w = L_Y^T v   (X = L_X L_X^T: X_inv holds L_X;  Y = L_Y L_Y^T).
# The stock code forms the full (d m) x (d m) pairing matrices V^T X^-1 V and V^T Y V in Arb for every block
# (d = rank per sample, m = samples: 11037^2 entries at (16,16)), which costs ~O((dm)^2) per block regardless
# of its size s. Here U and W (s x dm) are formed once, and for every pair of sample tiles (upper triangle)
# the small products U_P^T U_Q and W_P^T W_Q are formed in a task and folded into S directly.
# <A_p, Y> (A_Y, needed by the residuals) is the squared column norm of W, stored in the stock order.
using ClusteredLowRankSolver
include(joinpath(@__DIR__, "fastmul.jl"))
const LT = ClusteredLowRankSolver
const LTA = LT.Arblib
const LTM = LTA.ArbRefMatrix

function lr_symmetric(D)
    isempty(D) && return false
    for (k, A) in D
        A isa LT.LowRankMat || return false
        (A.vs === A.ws || A.vs == A.ws) || return false
        return true   # the blocks are built uniformly; checking one entry suffices
    end
end

const _ARBSZ = sizeof(LTA.arb_struct)
_arbptr(x::LTA.Arb) = Ptr{LTA.arb_struct}(pointer_from_objref(x.arb))   # x must be GC.@preserve'd at the call site
@inline _arb_addmul(z, x, y, prec) = ccall((:arb_addmul, LTA.libarb), Cvoid, (Ptr{LTA.arb_struct}, Ptr{LTA.arb_struct}, Ptr{LTA.arb_struct}, Int), z, x, y, prec)
@inline _arb_mul(z, x, y, prec) = ccall((:arb_mul, LTA.libarb), Cvoid, (Ptr{LTA.arb_struct}, Ptr{LTA.arb_struct}, Ptr{LTA.arb_struct}, Int), z, x, y, prec)

function _win(A, r1, c1, r2, c2)
    W = LTM(0, 0, prec=precision(A))
    LTA.window_init!(W, A, r1, c1, r2, c2)
    W
end

function lowrank_tiled!(S, A_Y, sdp, X_inv, Y, j, l, prec; T=0)
    # tile side: from K5_LRTILE, else about m / sqrt(6 nthreads) clamped to [64, 256], so that a block has >= ~3
    # tile pairs per thread (T = 256 left most of 192 threads idle at m ~ 3700: 120 pairs)
    T = T > 0 ? T : haskey(ENV, "K5_LRTILE") ? parse(Int, ENV["K5_LRTILE"]) : 0
    _lg = haskey(ENV, "K5_LRLOG"); _t = time()
    D = sdp.A[j][l][1, 1]
    keysD = collect(keys(D))          # Dict order: the order of A_Y entries in the stock code
    Ps = sort(keysD)
    m = length(Ps)
    m == 0 && return
    s = size(Y.blocks[j].blocks[l], 1)
    rk = [length(D[p].lambda) for p in Ps]
    cstart = zeros(Int, m)            # 0-based first column of sample a
    for a = 2:m; cstart[a] = cstart[a-1] + rk[a-1]; end
    ncol = cstart[m] + rk[m]
    V = LTM(s, ncol, prec=prec)
    lam = Vector{LTA.Arb}(undef, ncol)    # concrete element types throughout: the tile loop below must not box
    lone = falses(ncol)
    Threads.@threads for a = 1:m
        p = Ps[a]
        for r = 1:rk[a]
            c = cstart[a] + r
            vr = D[p].vs[r]
            for i = 1:s
                LTA.set!(V[i, c], vr[i])
            end
            lam[c] = D[p].lambda[r]
            lone[c] = LTA.isone(D[p].lambda[r])
        end
    end
    _lg && (print("LR l=$l s=$(size(Y.blocks[j].blocks[l],1)) ncol=$ncol V=$(round(time()-_t,digits=2)) "); _t = time())
    # U = L_X^-1 V (threaded over column chunks), W = L_Y^T V
    LX = X_inv.blocks[j].blocks[l]
    LY = LTM(s, s, prec=prec)
    LTA.set!(LY, Y.blocks[j].blocks[l])
    LT.approx_cholesky!(LY) == 0 && error("lowrank_tiled!: Cholesky of the Y block failed (block $l)")
    LTA.get_mid!(LY, LY)
    for i = 1:s, k = i+1:s; LTA.zero!(LY[i, k]); end
    LYt = LTM(s, s, prec=prec); LTA.transpose!(LYt, LY)
    U = LTM(s, ncol, prec=prec); W = LTM(s, ncol, prec=prec)
    chunks = collect(Iterators.partition(0:ncol-1, max(64, cld(ncol, 4 * Threads.nthreads()))))
    Threads.@threads for ch in chunks
        c1 = first(ch); c2 = last(ch) + 1
        Vc = _win(V, 0, c1, s, c2)
        Uc = LTM(s, c2 - c1, prec=prec)
        LTA.approx_solve_tril!(Uc, LX, Vc, 0)
        LTA.set!(_win(U, 0, c1, s, c2), Uc)
        Wc = LTM(s, c2 - c1, prec=prec)
        Main.k5mul!(Wc, LYt, Vc, prec=prec)
        LTA.set!(_win(W, 0, c1, s, c2), Wc)
    end
    LTA.get_mid!(U, U); LTA.get_mid!(W, W)
    _lg && (print("UW=$(round(time()-_t,digits=2)) "); _t = time())
    # A_Y: v^T Y v = |w|^2, in the stock order (for p in keys(D), for rank)
    colof = Dict(p => cstart[a] for (a, p) in enumerate(Ps))
    AYb = A_Y[j][l][1, 1]
    order = Tuple{Int,Int}[]
    for p in keysD, r = 1:length(D[p].lambda)
        push!(order, (length(order) + 1, colof[p] + r))
    end
    Threads.@threads for (idx, c) in order
        e = AYb[idx, 1]
        LTA.zero!(e)
        for i = 1:s
            LTA.addmul!(e, W[i, c], W[i, c])
        end
        LTA.get_mid!(e, e)
    end
    _lg && (print("AY=$(round(time()-_t,digits=2)) "); _t = time())
    # tile pairs of samples (upper triangle), each in its own task
    T == 0 && (T = clamp(m ÷ isqrt(6 * Threads.nthreads()), 64, 256))
    tiles = collect(Iterators.partition(1:m, T))
    tpairs = [(a, b) for b in eachindex(tiles) for a in 1:b]
    Sj = S[j]
    # tile buffers come from a pool (one set per concurrent task), sized for the largest tile of this block,
    # so the loop does not allocate (allocation churn triggered repeated full GCs on large problems)
    cmax = maximum(cstart[last(t)] + rk[last(t)] - cstart[first(t)] for t in tiles)
    npool = min(length(tpairs), Threads.nthreads())
    pool = Channel{Tuple{LTM,LTM,LTM,LTM,LTA.Arb}}(npool)
    for _ = 1:npool
        put!(pool, (LTM(cmax, s, prec=prec), LTM(cmax, s, prec=prec), LTM(cmax, cmax, prec=prec), LTM(cmax, cmax, prec=prec), LTA.Arb(prec=prec)))
    end
    alg = Main.k5alg((:lr, j, l)); _ttiles = time()
    Threads.@threads :dynamic for (ta, tb) in tpairs
        At = tiles[ta]; Bt = tiles[tb]
        ca1 = cstart[first(At)]; ca2 = cstart[last(At)] + rk[last(At)]
        cb1 = cstart[first(Bt)]; cb2 = cstart[last(Bt)] + rk[last(Bt)]
        bufs = take!(pool)
        UaT = _win(bufs[1], 0, 0, ca2 - ca1, s); LTA.transpose!(UaT, _win(U, 0, ca1, s, ca2))
        WaT = _win(bufs[2], 0, 0, ca2 - ca1, s); LTA.transpose!(WaT, _win(W, 0, ca1, s, ca2))
        BX = _win(bufs[3], 0, 0, ca2 - ca1, cb2 - cb1); BY = _win(bufs[4], 0, 0, ca2 - ca1, cb2 - cb1)
        Main.k5mul!(BX, UaT, _win(U, 0, cb1, s, cb2), alg, prec=prec)
        Main.k5mul!(BY, WaT, _win(W, 0, cb1, s, cb2), alg, prec=prec)
        tot = bufs[5]
        # entries are addressed through the row pointers of the arb_mat structs and updated with direct libarb
        # calls: A[i, j] costs a ccall (arb_mat_entry_ptr) plus a heap-allocated ArbRef per access
        GC.@preserve Sj BX BY tot lam begin
            rS = Sj.arb_mat.rows; rX = BX.arb_mat.rows; rY = BY.arb_mat.rows
            ptot = _arbptr(tot)
            @inbounds for b in Bt, a in At
                (ta == tb && a > b) && continue
                pe = unsafe_load(rS, Ps[a]) + (Ps[b] - 1) * _ARBSZ
                for r1 = 1:rk[a], r2 = 1:rk[b]
                    ca = cstart[a] + r1; cb = cstart[b] + r2
                    x = ca - ca1; y = cb - cb1
                    px = unsafe_load(rX, x) + (y - 1) * _ARBSZ; py = unsafe_load(rY, x) + (y - 1) * _ARBSZ
                    if lone[ca] && lone[cb]
                        _arb_addmul(pe, px, py, prec)
                    else
                        _arb_mul(ptot, _arbptr(lam[ca]), _arbptr(lam[cb]), prec)
                        _arb_mul(ptot, ptot, px, prec)
                        _arb_addmul(pe, ptot, py, prec)
                    end
                end
            end
        end
        put!(pool, bufs)
    end
    Main.k5alg_done!((:lr, j, l), alg, time() - _ttiles)
    _lg && println("tiles=$(round(time()-_t,digits=2)) alg=$(alg ? "classical" : "block")")
    return
end
