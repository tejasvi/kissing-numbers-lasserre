# Ozaki-split (exact DGEMM digit products, triple-double accumulation) Schur complement for
# ClusteredLowRankSolver 1.0.3: the compute_S_integrated! part of fast_S.jl from the level-2 "fast"
# pipeline (supplement of arXiv:2609.25120, used by ~/d4opt), lines before its fast_cholesky!; the
# trace_A/compute_weighted_A! overrides and fast_cholesky! are not used. Included INTO the
# ClusteredLowRankSolver module (Base.include(ClusteredLowRankSolver, file)) by k5_ozaki.jl.
# Digits/levels from ENV K5_OZ_K (default 8) and K5_OZ_L (default 8): error about 2^-(21 L - 10)
# relative to the column maxima of the Gram factors, limited by triple-double (~159 bits).
# Schur complement of ClusteredLowRankSolver 1.0.3 with its products done in Float64 BLAS,
# exactly, by splitting (the scheme of Ozaki, Ogita, Oishi and Rump), for the LLM24 programme.
# Evaluated into the module after patched_solver.jl and tiled_S.jl (whose function is renamed
# compute_S_integrated_tiled!).  With FAST[] = false the tiled version runs unchanged.
#
# Every contribution to S is a Gram matrix:
#   low-rank block:  <A_p, X^-1 A_q Y> = sum_{r,s} l_pr l_qs (w_pr . w_qs)(u_pr . u_qs),
#                    w = L_Y^T v,  u = L_X^-1 v   (Y = L_Y L_Y^T, X = L_X L_X^T),
#   high-rank block: <A_p, X^-1 A_q Y> = <G_p, G_q>,  G_p = L_X^-1 A_p L_Y.
# The vectors are computed in Arb, each column is written as 2^(e - K b) sum_k d_k 2^(b k) with
# integer digits |d_k| <= 2^(b-1) (+1 for the top one), and the dot products of digit
# vectors are formed by DGEMM, exactly (n 2^(2b) times the number of terms of a level stays
# below 2^53).  The top LEVELS levels of the digit expansion are kept and summed in triple-double,
# so each Gram entry has an error of about 2^-(b LEVELS - 10) times the product of the two
# column maxima; the Hadamard products and the sums over ranks are formed in triple-double.
using LinearAlgebra
const FAST = Ref(true)
const OZ_K = Ref(parse(Int, get(ENV, "K5_OZ_K", "8")))          # digits per entry
const OZ_B = Ref(21)         # bits per digit
const OZ_LEVELS = Ref(parse(Int, get(ENV, "K5_OZ_L", "8")))     # top levels kept
const OZ_TILE = Ref(1800)    # columns per tile
const FPROF = zeros(8)       # W, digits, gemm, combine+hadamard, highrank G, highrank gram, final add

@inline function _two_sum(a::Float64, b::Float64)
    s = a + b; bb = s - a; e = (a - (s - bb)) + (b - bb); return s, e
end
@inline function _qtwo_sum(a::Float64, b::Float64)
    s = a + b; e = b - (s - a); return s, e
end
@inline function _dd_add_d(hi::Float64, lo::Float64, b::Float64)
    s, e = _two_sum(hi, b); e += lo; return _qtwo_sum(s, e)
end
@inline function _dd_add(ah::Float64, al::Float64, bh::Float64, bl::Float64)
    s, e = _two_sum(ah, bh); t, f = _two_sum(al, bl); e += t; s, e = _qtwo_sum(s, e); e += f; return _qtwo_sum(s, e)
end
@inline function _dd_mul(ah::Float64, al::Float64, bh::Float64, bl::Float64)
    p = ah * bh; e = fma(ah, bh, -p); e += ah * bl + al * bh; return _qtwo_sum(p, e)
end

# digits of the columns of the Arb matrix W (n x c): D[:, q, k] is digit k-1, E[q] the scale
function oz_digits(W::ArbRefMatrix; K=OZ_K[], b=OZ_B[])
    @assert b == 21 && 4 <= K <= 9
    n, c = size(W)
    D = zeros(Float64, n, c, K); E = zeros(Int, c)
    mask = (Int128(1) << b) - 1; half = Int128(1) << (b - 1); full = Int128(1) << b
    Threads.@threads for q = 1:c
        t = AL.Arb(prec=precision(W)); u = AL.Arb(prec=precision(W))
        h = Vector{Float64}(undef, n); l1 = Vector{Float64}(undef, n); l2 = Vector{Float64}(undef, n)
        for i = 1:n
            x = W[i, q]
            h[i] = Float64(x); Arblib.set!(u, h[i]); Arblib.sub!(t, x, u)
            l1[i] = Float64(t); Arblib.set!(u, l1[i]); Arblib.sub!(t, t, u)
            l2[i] = Float64(t)
        end
        mx = maximum(abs, h)
        mx == 0 && continue
        e = exponent(mx) + 1
        E[q] = e
        sh = K * b - e
        for i = 1:n
            # the scaled value a + cc + g (exact, |.| < 2^(K b)) as H 2^63 + Lo with 0 <= Lo < 2^63
            a = ldexp(h[i], sh); cc = ldexp(l1[i], sh); g = ldexp(l2[i], sh)
            ah = round(ldexp(a, -63)); ch = round(ldexp(cc, -63))
            H = Int128(ah) + Int128(ch)
            Lo = Int128(round(a - ldexp(ah, 63))) + Int128(round(cc - ldexp(ch, 63))) + Int128(round(g))
            carry = Lo >> 63; Lo -= carry << 63; H += carry
            for k = 1:3
                d = Lo & mask
                d >= half && (d -= full)
                D[i, q, k] = Float64(d)
                Lo = (Lo - d) >> b
            end
            H += Lo
            for k = 4:K-1
                d = H & mask
                d >= half && (d -= full)
                D[i, q, k] = Float64(d)
                H = (H - d) >> b
            end
            D[i, q, K] = Float64(H)
        end
    end
    return D, E
end

# levels T[1..L] (T[1] the top) of the Gram products of the digit columns ca (of Da) and cb (of Db)
function oz_levels!(T, D1, ca, D2, cb; K=OZ_K[], L=OZ_LEVELS[])
    @assert size(D1, 3) == K
    @assert size(D1, 1) * 2.0^(2 * OZ_B[]) * (1 + (K - 2) / 4) * 1.01 < 2.0^53 "inner dimension too large for exact digit products"
    for li = 1:L
        s = 2 * (K - 1) - (li - 1)
        Tl = T[li]
        first = true
        for i = max(0, s - (K - 1)):min(K - 1, s)
            j = s - i
            BLAS.gemm!('T', 'N', 1.0, view(D1, :, ca, i + 1), view(D2, :, cb, j + 1), first ? 0.0 : 1.0, Tl)
            first = false
        end
    end
end

# entry (x, y) of the Gram matrix from the levels, as double-double, scaled by 2^(ex + ey - 2b)
@inline function oz_entry(T, x, y, ex, ey; L=OZ_LEVELS[], b=OZ_B[])
    hi = T[1][x, y]; lo = 0.0
    for li = 2:L
        hi, lo = _dd_add_d(hi, lo, ldexp(T[li][x, y], -b * (li - 1)))
    end
    sc = ex + ey - 2b
    return ldexp(hi, sc), ldexp(lo, sc)
end

function arb_dd(x)
    h = Float64(x); t = AL.Arb(x, prec=precision(x)); Arblib.sub!(t, t, AL.Arb(h, prec=precision(x)))
    return h, Float64(t)
end

# sum the levels of a tile in place: T[1] becomes the high part, T[2] the low part (unscaled)
function combine_levels!(Tv, L::Int, b::Int)
    T1 = Tv[1]; T2 = Tv[2]
    nr, nc = size(T1)
    Threads.@threads for y = 1:nc
        @inbounds for x = 1:nr
            hi = T1[x, y]; lo = 0.0
            for li = 2:L
                hi, lo = _dd_add_d(hi, lo, ldexp(Tv[li][x, y], -b * (li - 1)))
            end
            T1[x, y] = hi; T2[x, y] = lo
        end
    end
end

# Triple-double arithmetic (about 159 bits).  At (14,16) the Schur matrix is so ill-conditioned
# (Cholesky pivots down to 1e-26 of the diagonal at the first iteration) that double-double
# sums of the levels and double-double Hadamard products lose it; everything below that is
# summed or multiplied after the exact digit products is done in triple-double.
@inline function _two_prod(a::Float64, b::Float64)
    p = a * b; return p, fma(a, b, -p)
end
@inline function _renorm3(s0::Float64, s1::Float64, s2::Float64)
    s1, s2 = _two_sum(s1, s2); s0, s1 = _two_sum(s0, s1); s1, s2 = _two_sum(s1, s2)
    return s0, s1, s2
end
@inline function _td_add_d(a0::Float64, a1::Float64, a2::Float64, x::Float64)
    s0, e = _two_sum(a0, x); s1, e = _two_sum(a1, e)
    return _renorm3(s0, s1, a2 + e)
end
@inline function _td_add(a0::Float64, a1::Float64, a2::Float64, b0::Float64, b1::Float64, b2::Float64)
    s0, t0 = _two_sum(a0, b0); s1, t1 = _two_sum(a1, b1); s1, t0 = _two_sum(s1, t0)
    return _renorm3(s0, s1, (a2 + b2) + (t1 + t0))
end
@inline function _td_mul(a0::Float64, a1::Float64, a2::Float64, b0::Float64, b1::Float64, b2::Float64)
    p0, q0 = _two_prod(a0, b0); p1, q1 = _two_prod(a0, b1); p2, q2 = _two_prod(a1, b0)
    s1, e1 = _two_sum(q0, p1); s1, e2 = _two_sum(s1, p2)
    return _renorm3(p0, s1, ((e1 + e2) + (q1 + q2)) + ((a0 * b2 + a2 * b0) + a1 * b1))
end

# sum the levels of a tile in place: T[1], T[2], T[3] become the triple-double value (unscaled)
function combine_levels_td!(Tv, L::Int, b::Int)
    @assert L >= 3
    T1 = Tv[1]; T2 = Tv[2]; T3 = Tv[3]
    nr, nc = size(T1)
    f = [ldexp(1.0, -b * (li - 1)) for li = 1:L]
    Threads.@threads for y = 1:nc
        @inbounds for x = 1:nr
            a0 = T1[x, y]; a1 = 0.0; a2 = 0.0
            for li = 2:L
                a0, a1, a2 = _td_add_d(a0, a1, a2, Tv[li][x, y] * f[li])
            end
            T1[x, y] = a0; T2[x, y] = a1; T3[x, y] = a2
        end
    end
end

function arb_td(x)
    p = precision(x)
    h = Float64(x); t = AL.Arb(x, prec=p); Arblib.sub!(t, t, AL.Arb(h, prec=p))
    m = Float64(t); Arblib.sub!(t, t, AL.Arb(m, prec=p))
    return h, m, Float64(t)
end

# Arb value of a triple-double
function td_arb!(v, a0::Float64, a1::Float64, a2::Float64, tmp)
    Arblib.set!(v, a0)
    Arblib.set!(tmp, a1); Arblib.add!(v, v, tmp)
    Arblib.set!(tmp, a2); Arblib.add!(v, v, tmp)
    return v
end

# digits and levels: the exact products need n 2^(2b) (1 + (K-2)/4) < 2^53
inner_max(K) = floor(Int, 2.0^(53 - 2 * OZ_B[]) / ((1 + (K - 2) / 4) * 1.01))

# Gram entries of one tile pair of a high-rank block, added to S (upper triangle)
function hr_tile!(S0::Matrix{Float64}, S1::Matrix{Float64}, S2::Matrix{Float64}, H0, H1, H2, E::Vector{Int}, ps::Vector{Int},
                  ta::UnitRange{Int}, tb::UnitRange{Int}, same::Bool, b::Int)
    Threads.@threads for y in eachindex(tb)
        q = ps[tb[y]]; ey = E[tb[y]]
        @inbounds for x in eachindex(ta)
            (same && y < x) && break
            p = ps[ta[x]]
            sc = E[ta[x]] + ey - 2b
            pp, qq = p <= q ? (p, q) : (q, p)
            S0[pp, qq], S1[pp, qq], S2[pp, qq] = _td_add(S0[pp, qq], S1[pp, qq], S2[pp, qq],
                                                         ldexp(H0[x, y], sc), ldexp(H1[x, y], sc), ldexp(H2[x, y], sc))
        end
    end
end

# Hadamard products of one tile pair of a low-rank block, summed over the ranks and added to S
function lr_tile!(S0::Matrix{Float64}, S1::Matrix{Float64}, S2::Matrix{Float64}, Y0, Y1, Y2, X0, X1, X2,
                  EY::Vector{Int}, EX::Vector{Int}, l0::Vector{Float64}, l1::Vector{Float64}, l2::Vector{Float64},
                  ps::Vector{Int}, pstart::Vector{Int}, pnr::Vector{Int},
                  ia::UnitRange{Int}, ib::UnitRange{Int}, a0::Int, b0::Int, same::Bool, b::Int)
    Threads.@threads for iq in ib
        q = ps[iq]; sq = pstart[iq]; nq = pnr[iq]
        @inbounds for ip in ia
            (same && ip > iq) && break
            p = ps[ip]; sp = pstart[ip]; np_ = pnr[ip]
            c0 = 0.0; c1 = 0.0; c2 = 0.0
            for r2 = 1:nq, r1 = 1:np_
                x = sp + r1; y = sq + r2
                scy = EY[x] + EY[y] - 2b; scx = EX[x] + EX[y] - 2b
                xa = x - a0; yb = y - b0
                t0, t1, t2 = _td_mul(ldexp(Y0[xa, yb], scy), ldexp(Y1[xa, yb], scy), ldexp(Y2[xa, yb], scy),
                                     ldexp(X0[xa, yb], scx), ldexp(X1[xa, yb], scx), ldexp(X2[xa, yb], scx))
                m0, m1, m2 = _td_mul(l0[x], l1[x], l2[x], l0[y], l1[y], l2[y])
                t0, t1, t2 = _td_mul(t0, t1, t2, m0, m1, m2)
                c0, c1, c2 = _td_add(c0, c1, c2, t0, t1, t2)
            end
            pp, qq = p <= q ? (p, q) : (q, p)
            S0[pp, qq], S1[pp, qq], S2[pp, qq] = _td_add(S0[pp, qq], S1[pp, qq], S2[pp, qq], c0, c1, c2)
        end
    end
end

function lower_cholesky(Yl::ArbRefMatrix, prec)
    n = size(Yl, 1)
    LY = ArbRefMatrix(n, n, prec=prec)
    Arblib.set!(LY, Yl); approx_cholesky!(LY); Arblib.get_mid!(LY, LY)
    for i = 1:n, k = i+1:n; Arblib.zero!(LY[i, k]); end
    return LY
end

function fast_highrank!(S3, Adict, Yl::ArbRefMatrix, LX::ArbRefMatrix, prec::Int, L::Int)
    t0 = time()
    n = size(Yl, 1)
    LY = lower_cholesky(Yl, prec)
    ps = sort(collect(keys(Adict))); np = length(ps)
    mats = [Adict[p]::ArbRefMatrix for p in ps]
    Gt = ArbRefMatrix(n * n, np, prec=prec)
    Threads.@threads for ip = 1:np
        T1 = ArbRefMatrix(n, n, prec=prec); T2 = ArbRefMatrix(n, n, prec=prec)
        Arblib.approx_solve_tril!(T1, LX, mats[ip], 0)
        Arblib.approx_mul!(T2, T1, LY)
        for a = 1:n, bb = 1:n
            Arblib.set!(Gt[(bb - 1) * n + a, ip], T2[a, bb])
        end
    end
    FPROF[5] += time() - t0; t0 = time()
    D, E = oz_digits(Gt); Gt = nothing
    FPROF[2] += time() - t0
    tiles = collect(Iterators.partition(1:np, OZ_TILE[]))
    T = [zeros(Float64, OZ_TILE[], OZ_TILE[]) for _ = 1:L]
    for bi in eachindex(tiles), ai = 1:bi
        ta = tiles[ai]; tb = tiles[bi]
        t0 = time()
        Tv = [view(T[li], 1:length(ta), 1:length(tb)) for li = 1:L]
        oz_levels!(Tv, D, ta, D, tb)
        FPROF[3] += time() - t0; t0 = time()
        combine_levels_td!(Tv, L, OZ_B[])
        hr_tile!(S3[1], S3[2], S3[3], Tv[1], Tv[2], Tv[3], E, ps, ta, tb, ai == bi, OZ_B[])
        FPROF[6] += time() - t0
    end
end

function fast_lowrank!(S3, AYb::ArbRefMatrix, Adict, Yl::ArbRefMatrix, LX::ArbRefMatrix,
                       Rv::ArbRefMatrix, ptrR, prec::Int, L::Int)
    t0 = time()
    n = size(Yl, 1)
    LY = lower_cholesky(Yl, prec)
    kp = collect(keys(Adict))
    lams = Dict{Int, Vector{AL.Arb}}(p => (Adict[p]::LowRankMat).lambda for p in kp)
    offs = Dict{Int,Int}(); idx = 0
    for p in kp; offs[p] = idx; idx += length(lams[p]); end          # the order A_Y uses
    ps = sort(kp); nps = length(ps)
    pnr = [length(lams[p]) for p in ps]
    pstart = zeros(Int, nps); col = 0
    for i = 1:nps; pstart[i] = col; col += pnr[i]; end
    ctot = col
    V = ArbRefMatrix(n, ctot, prec=prec)
    l0 = zeros(ctot); l1 = zeros(ctot); l2 = zeros(ctot)
    for i = 1:nps, r = 1:pnr[i]
        c = pstart[i] + r; src = ptrR[(1, ps[i], r)]
        for k = 1:n; Arblib.set!(V[k, c], Rv[k, src]); end
        l0[c], l1[c], l2[c] = arb_td(lams[ps[i]][r])
    end
    LYt = ArbRefMatrix(n, n, prec=prec); Arblib.transpose!(LYt, LY)
    WY = ArbRefMatrix(n, ctot, prec=prec); WX = ArbRefMatrix(n, ctot, prec=prec)
    matmul_threaded!(WY, LYt, V, prec=prec)
    Arblib.approx_solve_tril!(WX, LX, V, 0)
    Arblib.get_mid!(WY, WY); Arblib.get_mid!(WX, WX)
    V = nothing
    FPROF[1] += time() - t0; t0 = time()
    DY, EY = oz_digits(WY); WY = nothing
    DX, EX = oz_digits(WX); WX = nothing
    FPROF[2] += time() - t0
    R = maximum(pnr)
    tiles = collect(Iterators.partition(1:nps, max(1, div(OZ_TILE[], R))))
    cr = [pstart[first(t)]+1:pstart[last(t)]+pnr[last(t)] for t in tiles]
    TY = [zeros(Float64, OZ_TILE[] + R, OZ_TILE[] + R) for _ = 1:L]
    TX = [zeros(Float64, OZ_TILE[] + R, OZ_TILE[] + R) for _ = 1:L]
    tmp = AL.Arb(prec=prec)
    for bi in eachindex(tiles), ai = 1:bi
        ca = cr[ai]; cb = cr[bi]
        t0 = time()
        TYv = [view(TY[li], 1:length(ca), 1:length(cb)) for li = 1:L]
        TXv = [view(TX[li], 1:length(ca), 1:length(cb)) for li = 1:L]
        oz_levels!(TYv, DY, ca, DY, cb)
        oz_levels!(TXv, DX, ca, DX, cb)
        FPROF[3] += time() - t0; t0 = time()
        a0 = first(ca) - 1; b0 = first(cb) - 1
        combine_levels_td!(TYv, L, OZ_B[]); combine_levels_td!(TXv, L, OZ_B[])
        lr_tile!(S3[1], S3[2], S3[3], TYv[1], TYv[2], TYv[3], TXv[1], TXv[2], TXv[3], EY, EX, l0, l1, l2,
                 ps, pstart, pnr, tiles[ai], tiles[bi], a0, b0, ai == bi, OZ_B[])
        if ai == bi   # <A_p, Y> needs the diagonal of V^T Y V
            for i in tiles[ai], r = 1:pnr[i]
                x = pstart[i] + r
                sc = 2 * EY[x] - 2 * OZ_B[]
                v = AYb[offs[ps[i]] + r, 1]
                td_arb!(v, ldexp(TYv[1][x - a0, x - b0], sc), ldexp(TYv[2][x - a0, x - b0], sc), ldexp(TYv[3][x - a0, x - b0], sc), tmp)
            end
        end
        FPROF[4] += time() - t0
    end
end

function compute_S_integrated!(S,sdp,A_Y, X_inv, Y,bilinear_pairings_Y, bilinear_pairings_Xinv, leftvecs,rightvecs,pointers_left,pointers_right,high_ranks, tempX, temppart_r;matmul_prec=precision(S[1]))
    prec = precision(Y)
    L = OZ_LEVELS[]
    for j in eachindex(sdp.A)
        Arblib.zero!(S[j])
        m = size(S[j], 1)
        S3 = (zeros(Float64, m, m), zeros(Float64, m, m), zeros(Float64, m, m))
        for l in eachindex(sdp.A[j])
            @assert size(sdp.A[j][l]) == (1,1)
            if high_ranks[j][l]
                fast_highrank!(S3, sdp.A[j][l][1,1], Y.blocks[j].blocks[l], X_inv.blocks[j].blocks[l], prec, L)
            else
                fast_lowrank!(S3, A_Y[j][l][1,1], sdp.A[j][l][1,1], Y.blocks[j].blocks[l], X_inv.blocks[j].blocks[l],
                              rightvecs[j][l][1], pointers_right[j][l][1], prec, L)
            end
        end
        t0 = time()
        Sj = S[j]; S0, S1, S2 = S3
        Threads.@threads for q = 1:m
            tmp = AL.Arb(prec=prec)
            for p = 1:q
                (S0[p, q] == 0 && S1[p, q] == 0 && S2[p, q] == 0) && continue
                td_arb!(Sj[p, q], S0[p, q], S1[p, q], S2[p, q], tmp)
            end
        end
        FPROF[7] += time() - t0
        symmetric!(S[j])
        Arblib.get_mid!(S[j], S[j])
    end
    GC.gc(false)
end

