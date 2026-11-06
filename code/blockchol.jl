# Blocked, multithreaded approximate Cholesky for ArbRefMatrix (drop-in for ClusteredLowRankSolver.approx_cholesky!)
using ClusteredLowRankSolver
include(joinpath(@__DIR__, "fastmul.jl"))
const Arblib = ClusteredLowRankSolver.Arblib
const ArbRefMatrix = Arblib.ArbRefMatrix
const CLRS = ClusteredLowRankSolver

function _win(A, r1, c1, r2, c2)   # 0-based half-open window [r1,r2) x [c1,c2)
    W = ArbRefMatrix(0, 0, prec=precision(A))
    Arblib.window_init!(W, A, r1, c1, r2, c2)
    W
end

const PIVLOG = haskey(ENV, "K5_PIVLOG")
function blocked_cholesky!(A::ArbRefMatrix; prec=precision(A), bs=256, nt=Threads.nthreads())
    n = size(A, 1)
    size(A, 1) == size(A, 2) || error("The matrix must be square")
    Arblib.get_mid!(A, A)
    dg0 = PIVLOG ? [Float64(A[i, i]) for i = 1:n] : Float64[]
    k = 0
    while k < n
        b = min(bs, n - k)
        D = _win(A, k, k, k + b, k + b)
        Dc = ArbRefMatrix(b, b, prec=prec); Arblib.set!(Dc, D)
        ok = naive_cholesky!(Dc; prec=prec)
        ok == 0 && return 0
        Arblib.set!(D, Dc)
        m = n - k - b
        if m > 0
            P = _win(A, k + b, k, n, k + b)            # m x b panel
            PT = ArbRefMatrix(b, m, prec=prec)
            Arblib.transpose!(PT, P)
            # solve Dc * X = PT, in column chunks over threads
            chunks = collect(Iterators.partition(0:m-1, cld(m, nt)))
            Threads.@threads for ch in chunks
                c1 = first(ch); c2 = last(ch) + 1
                B = _win(PT, 0, c1, b, c2)
                X = ArbRefMatrix(b, c2 - c1, prec=prec)
                Arblib.approx_solve_tril!(X, Dc, B, 0, prec=prec)
                Arblib.set!(B, X)
            end
            Arblib.transpose!(P, PT)
            # trailing update A22 -= P * P^T, only the lower block-triangle, over (row-block, col-block) tiles
            tb = max(64, cld(m, 2 * isqrt(nt) + 1))
            tiles = [(r, c) for r in 0:tb:m-1 for c in 0:tb:r]
            Threads.@threads for (r, c) in tiles
                r2 = min(r + tb, m); c2 = min(c + tb, m)
                Pr = _win(P, r, 0, r2, b)
                PTc = _win(PT, 0, c, b, c2)
                T = ArbRefMatrix(r2 - r, c2 - c, prec=prec)
                Main.k5mul!(T, Pr, PTc, prec=prec)
                A22 = _win(A, k + b + r, k + b + c, k + b + r2, k + b + c2)
                Arblib.sub!(A22, A22, T)
                Arblib.get_mid!(A22, A22)
            end
        end
        k += b
    end
    for j = 2:n, i = 1:j-1
        A[i, j] = 0
    end
    if PIVLOG
        r = [Float64(A[i, i])^2 / dg0[i] for i = 1:n]
        k = argmin(r)
        println("PIVOT n=$n min pivot^2/diag = $(r[k]) at row $k; log10 quantiles: ", round.(log10.(sort(r)[[1, max(1, n ÷ 100), max(1, n ÷ 10), n ÷ 2]]), digits=1)); flush(stdout)
    end
    return 1
end

# copy of ClusteredLowRankSolver's naive approx_cholesky! (tools.jl), used for small blocks
function naive_cholesky!(A::ArbRefMatrix; prec=precision(A))
    n = size(A, 1)
    Arblib.get_mid!(A, A)
    for i = 1:n
        for j = 1:i-1
            for k = 1:j-1
                CLRS.approx_submul!(A[i,j], A[i,k], A[j,k], prec)
            end
            CLRS.approx_div!(A[i,j], A[i,j], A[j,j], prec)
        end
        for k = 1:i-1
            CLRS.approx_submul!(A[i,i], A[i,k], A[i,k], prec)
        end
        if Arblib.is_positive(A[i,i]) == 0
            return 0
        end
        CLRS.approx_sqrt!(A[i,i], A[i,i], prec)
    end
    for i = 1:n, j = i+1:n
        A[i,j] = 0
    end
    return 1
end

@eval CLRS function approx_cholesky!(A::ArbRefMatrix; prec=precision(A))
    if size(A, 1) >= 600
        return Main.blocked_cholesky!(A; prec=prec)
    else
        return Main.naive_cholesky!(A; prec=prec)
    end
end

# ClusteredLowRankSolver.matmul_threaded! (used by compute_weighted_A!, i.e. the residuals and dX) split C into
# n windows and called the block product; its stock size test also never threaded below 256 bits. Here C is cut
# into >= 4n square-ish tiles (side >= 32), each a classical product (fastmul.jl) in its own task, so outputs that
# are small but have a long inner dimension (395 x 11037 x 395 in the low-rank branch) also spread over threads.
@eval CLRS function matmul_threaded!(C::T, A::T, B::T; n = Threads.nthreads(), prec=precision(C), opt::Int=0) where T<:Union{AL.ArbMatrix, ArbRefMatrix}
    @assert size(C,1) == size(A,1) && size(C,2) == size(B,2) && size(A,2) == size(B,1)
    m, k, p = size(A, 1), size(A, 2), size(B, 2)
    if n == 1 || m * k * p < 2_000_000 || m == 0 || p == 0
        return Main.k5mul!(C, A, B, prec=prec)
    end
    t = max(32, floor(Int, sqrt(m * p / (4n))))
    tiles = [(r, c) for r in 0:t:m-1 for c in 0:t:p-1]
    Threads.@threads :dynamic for (r, c) in tiles
        r2 = min(r + t, m); c2 = min(c + t, p)
        Main.k5mul!(Main._win(C, r, c, r2, c2), Main._win(A, r, 0, r2, k), Main._win(B, 0, c, k, c2), prec=prec)
    end
    return C
end
