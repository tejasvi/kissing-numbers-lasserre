# compute the zonal polynomials P(S) for one lambda and several dimensions ns (comma list);
# then delete the n-independent base files of that lambda (they are huge) unless KEEP_BASE=1
using LasserreSphericalCodes
const L = LasserreSphericalCodes
ns = [parse(Int, x) for x in split(ARGS[1], ",")]; l1 = parse(Int, ARGS[2]); l2 = parse(Int, ARGS[3])
lambda = [l1, l2]
ws = [k for k = 0:lambda[1]-lambda[2] if iseven(L.countels(lambda, 2, k))]
for n in ns
    t = @elapsed L.compute_PS(n, lambda, 2; ws=ws)
    println("DONE n=$n lambda=$lambda ws=$ws time=$(t)s"); flush(stdout)
end
if get(ENV, "KEEP_BASE", "0") != "1"
    for l1i = 1:length(ws), l2i = 1:l1i
        f = "cache/zonalstore/pol-2-" * strip(string(lambda)) * "-" * string(ws[l1i]) * "-" * string(ws[l2i]) * ".txt"
        isfile(f) && rm(f)
    end
    println("deleted base files for $lambda"); flush(stdout)
end
