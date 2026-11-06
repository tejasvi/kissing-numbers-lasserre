# Smallest eigenvalue of a real symmetric matrix in pure Julia (no BLAS/LAPACK): Householder reduction to
# tridiagonal form, then Sturm-sequence bisection. OpenBLAS in this Julia is built for 64 threads and aborts
# ("too many memory regions") once BLAS has been called from too many of the 192 solver threads; the step-length
# and centrality computations call this instead. O(n^3) with plain loops: ~0.2 s at n = 654.
function k5_eigmin!(A::Matrix{Float64})
    n = size(A, 1)
    n == 0 && return Inf
    n == 1 && return A[1, 1]
    d = zeros(n); e = zeros(n - 1)
    v = zeros(n); p = zeros(n)
    @inbounds for k = 1:n-2
        # Householder vector for column k below the diagonal (lower triangle of A is used)
        sc = 0.0
        for i = k+1:n; sc = max(sc, abs(A[i, k])); end
        if sc == 0.0
            d[k] = A[k, k]; e[k] = 0.0; continue
        end
        s2 = 0.0
        for i = k+1:n; v[i] = A[i, k] / sc; s2 += v[i]^2; end
        nx = sqrt(s2)
        alpha = v[k+1] >= 0 ? -nx : nx
        v[k+1] -= alpha
        vn2 = 0.0
        for i = k+1:n; vn2 += v[i]^2; end
        d[k] = A[k, k]; e[k] = alpha * sc
        vn2 == 0.0 && continue
        beta = 2.0 / vn2
        # p = beta * A22 * v (A22 symmetric, lower triangle stored)
        for i = k+1:n; p[i] = 0.0; end
        for j = k+1:n
            vj = v[j]; acc = A[j, j] * vj
            for i = j+1:n
                a = A[i, j]
                acc += a * v[i]
                p[i] += a * vj
            end
            p[j] += acc
        end
        K = 0.0
        for i = k+1:n; p[i] *= beta; K += v[i] * p[i]; end
        K *= beta / 2
        for i = k+1:n; p[i] -= K * v[i]; end      # q = p - K v
        for j = k+1:n, i = j:n                     # A22 -= v q' + q v'
            A[i, j] -= v[i] * p[j] + p[i] * v[j]
        end
    end
    d[n-1] = A[n-1, n-1]; d[n] = A[n, n]; e[n-1] = A[n, n-1]
    # Sturm bisection for the smallest eigenvalue
    lo = Inf; hi = -Inf
    @inbounds for i = 1:n
        r = (i > 1 ? abs(e[i-1]) : 0.0) + (i < n ? abs(e[i]) : 0.0)
        lo = min(lo, d[i] - r); hi = max(hi, d[i] + r)
    end
    tiny = Base.eps(Float64) * max(abs(lo), abs(hi), 1e-300)   # Base.: solve_ckpt.jl has a global `eps`
    for _ = 1:200
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        cnt = 0; q = d[1] - mid
        q == 0.0 && (q = -tiny); q < 0 && (cnt += 1)
        @inbounds for i = 2:n
            q = d[i] - mid - e[i-1]^2 / q
            q == 0.0 && (q = -tiny); q < 0 && (cnt += 1)
        end
        cnt >= 1 ? (hi = mid) : (lo = mid)
        hi - lo <= 1e-14 * max(abs(lo), abs(hi)) && break
    end
    return (lo + hi) / 2
end
k5_eigmin(A::AbstractMatrix) = k5_eigmin!(Matrix{Float64}((A + A') / 2))
