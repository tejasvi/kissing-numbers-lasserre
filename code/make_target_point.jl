# A strictly feasible point of the target problem (bound Y_[0,0][1,1] fixed to tgt) built from two dual iterates
# of the OPTIMIZATION run, without a separate target solve:
#   Y_w = (1 - th) Y_A + th Y_B,  bound(A) > tgt > bound(B),  th = (bound(A) - tgt) / (bound(A) - bound(B)).
# Both iterates satisfy the SOS identities up to the solver's dual residual, so Y_w does too (the bound is linear),
# and Y_w is positive definite with lmin(Y_w) >= (1 - th) lmin(Y_A) + th lmin(Y_B). On the central path
# lmin(Y_A) grows like mu_A ~ (bound(A) - L2) / N while 1 - th = (tgt - bound(B)) / (bound(A) - bound(B)), so the
# interior push is about the same for every A; the residual weight of A, (1 - th) |d_A|, is smallest for an A far
# up the path whose residual is small relative to its bound. The rounding then only has to project Y_w onto the
# exact affine space (roundsol.jl with K5_RS_PRIMAL=0: the primal is not used, any iterate's primal will do).
#   julia make_target_point.jl A.jls B.jls tgt OUT.jls      (A, B: ckpt.jls / solution.jls of the optimization run)
using ClusteredLowRankSolver, Serialization, LinearAlgebra, Printf   # (BigFloat eigvals: GenericLinearAlgebra, loaded by CLRS)
include(joinpath(@__DIR__, "k5ser.jl"))
setprecision(BigFloat, 512)
A = deserialize(ARGS[1]); B = deserialize(ARGS[2]); tgt = BigFloat(parse(Rational{BigInt}, ARGS[3])); out = ARGS[4]
dA = A[2]; dB = B[2]
bA = BigFloat(dA.matrixvars[[0, 0]][1, 1]); bB = BigFloat(dB.matrixvars[[0, 0]][1, 1])
@printf("bound(A) = %.15e  bound(B) = %.15e  target = %.15e\n", bA, bB, tgt)
bA > tgt > bB || error("need bound(A) > target > bound(B)")
th = (bA - tgt) / (bA - bB)
@printf("1 - theta = %.6e\n", 1 - th)
mvs = Dict{Any,Matrix{BigFloat}}(k => (1 - th) .* BigFloat.(dA.matrixvars[k]) .+ th .* BigFloat.(dB.matrixvars[k])
                                 for k in keys(dB.matrixvars))
fvs = Dict{Any,BigFloat}(k => (1 - th) * BigFloat(dA.freevars[k]) + th * BigFloat(dB.freevars[k]) for k in keys(dB.freevars))
dW = ClusteredLowRankSolver.DualSolution{BigFloat}(dB.base_ring, mvs, fvs)
@printf("bound(W) - target = %.3e\n", mvs[[0, 0]][1, 1] - tgt)
serialize(out, (B[1], dW))
println("wrote $out")

# smallest eigenvalue of every block after diagonal scaling (D^-1/2 Y D^-1/2, unit diagonal): Float64 where it
# resolves it, else a Cholesky bracket by bisection (largest t = 10^-e, e in 12..60, with S - t I positive definite)
function scaledmin(M)
    s = size(M, 1)
    s == 0 && return Inf
    d = [M[i, i] > 0 ? 1 / sqrt(M[i, i]) : big(1) for i = 1:s]
    S = [d[i] * M[i, j] * d[j] for i = 1:s, j = 1:s]
    ev = eigvals(Symmetric(Float64.(S)))
    ev[1] > 1e-11 && return ev[1]
    pd(e) = isposdef(Symmetric(S - BigFloat(10)^(-e) * I))
    lo, hi = 12, 60          # smallest e in [12, 60] with S - 10^-e I positive definite (monotone in e)
    pd(hi) || return 0.0
    while lo < hi
        mid = (lo + hi) ÷ 2
        pd(mid) ? (hi = mid) : (lo = mid + 1)
    end
    return 10.0^(-hi)
end
for (name, ds) in (get(ENV, "K5_MTP_DIAG", "W") == "WAB" ? (("W", dW), ("A", dA), ("B", dB)) : (("W", dW),))   # A, B too with K5_MTP_DIAG=WAB
    ks = sort(collect(keys(ds.matrixvars)), by=k -> -size(ds.matrixvars[k], 1))
    vals = Vector{Float64}(undef, length(ks))
    t = @elapsed Threads.@threads :dynamic for i in eachindex(ks)
        vals[i] = scaledmin(BigFloat.(ds.matrixvars[ks[i]]))
    end
    o = sortperm(vals)
    println("$name: smallest eigenvalues of the diagonally scaled blocks ($(round(t, digits=1))s; Cholesky brackets are lower bounds):")
    for i in o[1:min(10, end)]
        @printf("  %-22s %5d  %.3e\n", string(ks[i]), size(ds.matrixvars[ks[i]], 1), vals[i])
    end
    flush(stdout)
end
