# Exact rounding + verification of a level-2 solution produced by solve_ckpt.jl with an objective target
# (feasibility version: the bound is fixed to tgt by a constraint), so that the long numerical solve runs in the
# checkpointed driver and only the rounding happens here:
#   julia roundsol.jl n lo cos d1 delta tgt SOLUTION.jls      e.g. 5 -1 1//2 16 16 4399//100 solution.jls
# Writes proofs/<n>_<floor(tgt)>/ and runs the verifier of arXiv:2404.18794 (LasserreSphericalCodes.verify).
# The rounding mirrors LasserreSphericalCodes.save_solution(n, Npoints, costheta, dzonal; obj, delta, d2).
using LasserreSphericalCodes, ClusteredLowRankSolver, Serialization
include(joinpath(@__DIR__, "k5ser.jl"))   # binary SDP cache format
include(joinpath(@__DIR__, "fastround.jl"))   # parallel no-kernel prescreen, identity transforms, FLINT-level projection helpers
# threaded FLINT (matrix products in the RREF mod p and the Dixon solve): needs NEMO_THREADED=1 at startup
if get(ENV, "NEMO_THREADED", "") == "1" && haskey(ENV, "K5_FLINT_THREADS")
    ClusteredLowRankSolver.Nemo.flint_set_num_threads(parse(Int, ENV["K5_FLINT_THREADS"]))
    println("FLINT threads: ", ENV["K5_FLINT_THREADS"]); flush(stdout)
end
const L = LasserreSphericalCodes
n = parse(Int, ARGS[1]); lo = parse(Rational{BigInt}, ARGS[2]); c = parse(Rational{BigInt}, ARGS[3])
d1 = parse(Int, ARGS[4]); dl = parse(Int, ARGS[5]); tgt = parse(Rational{BigInt}, ARGS[6]); solfile = ARGS[7]
lo == -1 || error("save_solution / verify assume the domain [-1, cos]")
setprecision(BigFloat, 256)
sol = deserialize(solfile)
primalsol, dualsol = sol[1], sol[2]
t0 = time()
problem = L.las2_problem(n, [-1, c], d1, sampled=false, delta=dl, d2=dl, obj=tgt)
println("exact problem built in $(round(time() - t0, digits=1))s"); flush(stdout)
degs = [-1, max(dl, d1) * ones(Int, 3)...]
bases = L.coefficient_matching_bases(degs)
R, x = L.polynomial_ring(L.QQ, 1)
push!(bases, [R(1)])     # the objective constraint
t1 = time()
# rounding settings from the environment (defaults as in round13.jl): K5_RS_KERR kernel_errbound, K5_RS_KRERR
# kernel_round_errbound, K5_RS_LLL=1 kernel_lll, K5_RS_REDUCE=0 no reduce_kernelvectors, K5_RS_PRIMAL=0
rs = RoundingSettings(kernel_errbound=parse(Float64, get(ENV, "K5_RS_KERR", "1e-6")),
    kernel_round_errbound=parse(Float64, get(ENV, "K5_RS_KRERR", "1e-15")),
    kernel_lll=get(ENV, "K5_RS_LLL", "0") == "1", reduce_kernelvectors=get(ENV, "K5_RS_REDUCE", "1") == "1",
    kernel_use_primal=get(ENV, "K5_RS_PRIMAL", "1") == "1",
    redundancyfactor=parse(Int, get(ENV, "K5_RS_REDUND", "10")),   # columns kept = factor x rows (stock 10; +2 per inconsistent retry)
    pseudo=get(ENV, "K5_RS_PSEUDO", "1") == "1")   # 0: exact square solve on the pivot columns instead of the min-norm pseudo-inverse
println("rounding settings: kernel_errbound=$(rs.kernel_errbound) kernel_round_errbound=$(rs.kernel_round_errbound) ",
        "kernel_lll=$(rs.kernel_lll) reduce=$(rs.reduce_kernelvectors) use_primal=$(rs.kernel_use_primal) redundancy=$(rs.redundancyfactor) pseudo=$(rs.pseudo)"); flush(stdout)
rs.kernel_use_primal || k5_prescreen!(dualsol, rs)
success, esol, transformations = exact_solution(problem, primalsol, dualsol, monomial_bases=bases,
    settings=rs, transformed=true)
println("ROUNDING success=$success in $(round(time() - t1, digits=1))s"); flush(stdout)
res = L.save_solution(success, problem, esol, transformations, n, tgt, c, d1, []; delta=dl, d2=dl)
folder = "proofs/$(n)_$(floor(Int, tgt))"
t2 = @elapsed verify(folder)
println("VERIFY $folder done in $(t2)s")
