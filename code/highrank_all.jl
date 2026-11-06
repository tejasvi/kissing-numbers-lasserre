# All high-rank (kernel) blocks of a cluster in one Gram product. For each block l and sample p,
#   G^l_p = L_X^-1 A_p L_Y  (X = L_X L_X^T: X_inv holds L_X;  Y = L_Y L_Y^T),   <A_p, X^-1 A_q Y> = <G^l_p, G^l_q>,
# so S_pq += sum_l <G^l_p, G^l_q> = <g_p, g_q> with g_p the concatenation of vec(G^l_p) over all high-rank
# blocks. The per-block version forms and adds an m x m matrix for each of ~80 blocks; here the stacked
# vectors are formed once (threaded over samples) and the Gram matrix is computed over tile pairs of samples
# (upper triangle only), each tile pair in its own task, and added to S once.
using ClusteredLowRankSolver
include(joinpath(@__DIR__, "fastmul.jl"))
const HA = ClusteredLowRankSolver
const HAA = HA.Arblib
const HAM = HAA.ArbRefMatrix

function _hwin(A, r1, c1, r2, c2)
    W = HAM(0, 0, prec=precision(A))
    HAA.window_init!(W, A, r1, c1, r2, c2)
    W
end

function highrank_all!(S, sdp, X_inv, Y, j, high_ranks, prec; T=0)
    # tile side: from K5_HRTILE, else about m / sqrt(6 nthreads) clamped to [64, 256], so that a block has >= ~3
    # tile pairs per thread (T = 256 left most of 192 threads idle at m ~ 3700: 120 pairs)
    T = T > 0 ? T : haskey(ENV, "K5_HRTILE") ? parse(Int, ENV["K5_HRTILE"]) : 0
    _lg = haskey(ENV, "K5_LRLOG"); _t = time()
    ls = [l for l in eachindex(sdp.A[j]) if high_ranks[j][l]]
    isempty(ls) && return
    Pset = Set{Int}()
    for l in ls; union!(Pset, keys(sdp.A[j][l][1, 1])); end
    Ps = sort(collect(Pset)); m = length(Ps)
    pos = Dict(p => a for (a, p) in enumerate(Ps))
    sz = [size(Y.blocks[j].blocks[l], 1) for l in ls]
    roff = zeros(Int, length(ls))
    for b = 2:length(ls); roff[b] = roff[b-1] + sz[b-1]^2; end
    nrows = roff[end] + sz[end]^2
    LYs = Vector{HAM}(undef, length(ls))
    Threads.@threads for b in eachindex(ls)
        l = ls[b]; s = sz[b]
        LY = HAM(s, s, prec=prec)
        HAA.set!(LY, Y.blocks[j].blocks[l])
        HA.approx_cholesky!(LY) == 0 && error("highrank_all!: Cholesky of the Y block failed (block $l)")
        HAA.get_mid!(LY, LY)
        for i = 1:s, k = i+1:s; HAA.zero!(LY[i, k]); end
        LYs[b] = LY
    end
    _lg && (print("HR blocks=$(length(ls)) m=$m nrows=$nrows LY=$(round(time()-_t,digits=2)) "); _t = time())
    GT = HAM(m, nrows, prec=prec)              # row a: g_{Ps[a]}
    # per-task scratch (max block size), windows of it per block: no allocation per sample and block
    smax = maximum(sz)
    chunks = collect(Iterators.partition(1:m, max(1, cld(m, 4 * Threads.nthreads()))))
    Threads.@threads :dynamic for ch in chunks
        T1buf = HAM(smax, smax, prec=prec); T2buf = HAM(smax, smax, prec=prec)
        T1s = [_hwin(T1buf, 0, 0, s, s) for s in sz]; T2s = [_hwin(T2buf, 0, 0, s, s) for s in sz]
        for a in ch
        p = Ps[a]
        for b in eachindex(ls)
            l = ls[b]; s = sz[b]
            Ap = get(sdp.A[j][l][1, 1], p, nothing)
            Ap === nothing && continue
            T1 = T1s[b]; T2 = T2s[b]
            HAA.approx_solve_tril!(T1, X_inv.blocks[j].blocks[l], Ap, 0)
            Main.k5mul!(T2, T1, LYs[b], prec=prec)
            o = roff[b]
            for k = 1:s, i = 1:s
                HAA.set!(GT[a, o + (k - 1) * s + i], T2[i, k])
            end
        end
        end
    end
    HAA.get_mid!(GT, GT)
    _lg && (print("GT=$(round(time()-_t,digits=2)) "); _t = time())
    G = HAM(nrows, m, prec=prec)
    HAA.transpose!(G, GT)
    _lg && (print("transpose=$(round(time()-_t,digits=2)) "); _t = time())
    T == 0 && (T = clamp(m ÷ isqrt(6 * Threads.nthreads()), 64, 256))
    tiles = collect(Iterators.partition(1:m, T))
    tpairs = [(a, b) for b in eachindex(tiles) for a in 1:b]
    Sj = S[j]
    npool = min(length(tpairs), Threads.nthreads())
    pool = Channel{HAM}(npool)
    for _ = 1:npool; put!(pool, HAM(T, T, prec=prec)); end
    alg = Main.k5alg((:hr, j, 0, nrows)); _tg = time()
    Threads.@threads :dynamic for (ta, tb) in tpairs
        At = tiles[ta]; Bt = tiles[tb]
        Cbuf = take!(pool)
        C = _hwin(Cbuf, 0, 0, length(At), length(Bt))
        Main.k5mul!(C, _hwin(GT, first(At) - 1, 0, last(At), nrows), _hwin(G, 0, first(Bt) - 1, nrows, last(Bt)), alg, prec=prec)
        for (y, b) in enumerate(Bt), (x, a) in enumerate(At)
            (ta == tb && a > b) && continue
            e = Sj[Ps[a], Ps[b]]
            HAA.add!(e, e, C[x, y])
        end
        put!(pool, Cbuf)
    end
    Main.k5alg_done!((:hr, j, 0, nrows), alg, time() - _tg)
    _lg && println("gram=$(round(time()-_t,digits=2)) alg=$(alg ? "classical" : "block")")
    return
end
