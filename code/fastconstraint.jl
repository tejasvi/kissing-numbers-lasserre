# Fast (multithreaded, numerical) construction of the level-2 constraints for the SOLVE phase.
# Replaces LasserreSphericalCodes.las2_constraint when sampled=true. The zonal-matrix entries are
# evaluated per sample point in BigFloat (512 bits), in parallel over samples, and stored as exact
# dyadic rationals rounded to 2^-300. The exact problem used for rounding (sampled=false) still goes
# through the original code (kept as las2_constraint_slow).
using LasserreSphericalCodes, Nemo, ClusteredLowRankSolver
const LSC = LasserreSphericalCodes

# keep a renamed copy of the original las2_constraint
let src = read(joinpath(pkgdir(LSC), "src", "las2.jl"), String)
    i = findfirst("function las2_constraint(", src)
    j = findnext("\nend\n", src, first(i))
    code = replace(src[first(i):last(j)], "function las2_constraint(" => "function las2_constraint_slow("; count=1)
    Base.eval(LSC, Meta.parse(code))
end

const ZPREC = 512
const QROUND = BigInt(2)^300

# parsed zonal polynomial terms: (c, ev, fa_half), cached per (n, lambda, i, j)
const ZTERMS = Dict{Any,Any}()
const ZLOCK = ReentrantLock()

function zterms(n, lambda, w1, w2)
    t = 2
    i = max(w1, w2); j = min(w1, w2)
    key = (n, Tuple(lambda), i, j)
    cached = lock(ZLOCK) do
        get(ZTERMS, key, nothing)
    end
    cached === nothing || return cached
    terms = _parse_zterms(n, lambda, i, j)
    lock(ZLOCK) do
        get!(ZTERMS, key, terms)
    end
end

function _parse_zterms(n, lambda, i, j)
    t = 2
    begin
        begin
            fn = "cache/zonalstore/$n/pol-$(n)-" * string(t) * "-" * strip(string(lambda)) * "-" * string(i) * "-" * string(j) * ".txt"
            isfile(fn) || error("missing zonal file $fn")
            row_sums = [LSC.countels(lambda, k, i) for k = 1:t]
            col_sums = [LSC.countels(lambda, k, j) for k = 1:t]
            terms = Tuple{BigFloat,NTuple{4,Int},NTuple{4,Int}}[]
            begin
                for l in eachline(fn)
                    els = split(l)
                    ev = [parse(Int, x) for x in els[1:end-1]]
                    c = BigFloat(parse(Rational{BigInt}, els[end]))
                    exp_mat = reshape(ev, t, t)
                    cur_rows = [sum(exp_mat[k, :]) for k = 1:t]
                    cur_cols = [sum(exp_mat[:, k]) for k = 1:t]
                    # factorsx = [e1, e2], factorsy = [e3, e4] (see scaling_factors)
                    fa_exp = [row_sums[1] - cur_rows[1], row_sums[2] - cur_rows[2], col_sums[1] - cur_cols[1], col_sums[2] - cur_cols[2]]
                    @assert all(iseven, fa_exp) && all(>=(0), fa_exp)
                    push!(terms, (c, Tuple(ev), Tuple(div.(fa_exp, 2))))
                end
            end
            terms
        end
    end
end

# numerical zonal entry at inner products u (6 numbers), entry (w1, w2)
function zeval(terms, w1, w2, u::Vector{BigFloat})
    if max(w1, w2) != w1
        u = [u[2], u[1], u[3], u[5], u[4], u[6]]
    end
    S = (u[3] + u[4] + u[5] + u[6], u[3] + u[4] - u[5] - u[6], u[3] - u[4] + u[5] - u[6], u[3] - u[4] - u[5] + u[6])
    P = (2 * (1 + u[1]), 2 * (1 - u[1]), 2 * (1 + u[2]), 2 * (1 - u[2]))
    maxe = 0
    for (c, ev, fh) in terms
        maxe = max(maxe, maximum(ev), maximum(fh))
    end
    Spow = [[s^e for e = 0:maxe] for s in S]
    Ppow = [[p^e for e = 0:maxe] for p in P]
    res = BigFloat(0)
    for (c, ev, fh) in terms
        res += c * Spow[1][ev[1]+1] * Spow[2][ev[2]+1] * Spow[3][ev[3]+1] * Spow[4][ev[4]+1] *
               Ppow[1][fh[1]+1] * Ppow[2][fh[2]+1] * Ppow[3][fh[3]+1] * Ppow[4][fh[4]+1]
    end
    res
end

toQQ(x::BigFloat) = QQ(BigInt(round(x * QROUND)), QROUND)

function gram_numeric(k, s)
    G = ones(BigFloat, k, k)
    idx = 1
    for a = 1:k, b = a+1:k
        G[a, b] = G[b, a] = BigFloat(Rational{BigInt}(s[idx]))
        idx += 1
    end
    G
end

function las2_constraint_fast(n, t, k, d_tensor, representatives, FF, all_samples,
        all_matrices, tuple_to_index, irreps, weven, blocksizes, verbose)
    subs = []
    for x in Iterators.flatten((LSC.IterTools.subsets(1:k, l) for l = 0:t))
        push!(subs, x)
    end
    oldprec = precision(BigFloat)
    setprecision(BigFloat, ZPREC)
    iterationtime = @elapsed begin
        samples = all_samples[k]
        ns = length(samples)
        R_sampled = SampledMPolyRing(FF, samples)
        matrices = all_matrices[k]
        needed = Set{Tuple{Vector{Int},Int,Int}}()
        for lambdaindex in eachindex(irreps), i1 in 0:2, i2 in 0:2
            for (_, k1) in LSC.admissible_tuples(irreps[lambdaindex], i1, weven[lambdaindex], d_tensor[lambdaindex]),
                (_, k2) in LSC.admissible_tuples(irreps[lambdaindex], i2, weven[lambdaindex], d_tensor[lambdaindex])
                push!(needed, (collect(irreps[lambdaindex]), max(k1, k2), min(k1, k2)))
            end
        end
        neededv = collect(needed)
        Threads.@threads for q in eachindex(neededv)
            lam, i, j = neededv[q]
            zterms(n, lam, i, j)
        end
        Gs = setprecision(BigFloat, ZPREC) do
            [gram_numeric(k, s) for s in samples]
        end
        # all (J1, J2) with J1 cup J2 = 1:k, and their numerical inner products per sample
        pairs = []
        for J1_index in 1:length(subs), J2_index in J1_index:length(subs)
            J1 = subs[J1_index]; J2 = subs[J2_index]
            sort(union(J1, J2)) == collect(1:k) || continue
            i1 = length(J1); i2 = length(J2)
            ipsv = setprecision(BigFloat, ZPREC) do
                map(Gs) do G
                    f11(a, b) = G[J1[min(a, i1)], J1[min(b, i1)]]
                    f22(a, b) = G[J2[min(a, i2)], J2[min(b, i2)]]
                    f12(a, b) = G[J1[min(a, i1)], J2[min(b, i2)]]
                    if i1 > 0 && i2 > 0
                        [f11(1, 2), f22(1, 2), f12(1, 1), f12(1, 2), f12(2, 1), f12(2, 2)]
                    else
                        v = ones(BigFloat, 6)
                        if i1 > 0
                            v[1] = f11(1, 2)
                        elseif i2 > 0
                            v[2] = f22(1, 2)
                        end
                        v
                    end
                end
            end
            push!(pairs, (J1, J2, i1, i2, ipsv))
        end
        for lambdaindex in eachindex(irreps)
            lambda = irreps[lambdaindex]
            if !(length(LSC.admissible_tuples(lambda, min(k, 2), weven[lambdaindex], d_tensor[lambdaindex])) > 0)
                continue
            end
            ts = blocksizes[lambdaindex]
            acc = Dict{Tuple{Int,Int},Vector{BigFloat}}()
            for (J1, J2, i1, i2, ipsv) in pairs
                adm1 = LSC.admissible_tuples(lambda, i1, weven[lambdaindex], d_tensor[lambdaindex])
                adm2 = LSC.admissible_tuples(lambda, i2, weven[lambdaindex], d_tensor[lambdaindex])
                (isempty(adm1) || isempty(adm2)) && continue
                kpairs = unique([(k1, k2) for (_, k1) in adm1 for (_, k2) in adm2])
                termsl = [zterms(n, lambda, k1, k2) for (k1, k2) in kpairs]
                valsl = [Vector{BigFloat}(undef, ns) for _ in kpairs]
                Threads.@threads for tsk = 1:(length(kpairs) * ns)
                    q = (tsk - 1) ÷ ns + 1; si = (tsk - 1) % ns + 1
                    k1, k2 = kpairs[q]
                    valsl[q][si] = zeval(termsl[q], k1, k2, ipsv[si])
                end
                Zval = Dict{Tuple{Int,Int},Vector{BigFloat}}(kpairs[q] => valsl[q] for q in eachindex(kpairs))
                f = J1 != J2 ? 1 : 1 // 2
                for (a, (j1, k1)) in enumerate(adm1), (b, (j2, k2)) in enumerate(adm2)
                    zk = Zval[(k1, k2)]
                    Za = tuple_to_index[lambdaindex][(i1, j1, k1)]
                    Zb = tuple_to_index[lambdaindex][(i2, j2, k2)]
                    va = get!(() -> zeros(BigFloat, ns), acc, (Za, Zb))
                    vb = get!(() -> zeros(BigFloat, ns), acc, (Zb, Za))
                    Threads.@threads for si = 1:ns
                        z = f * (ipsv[si][1]^j1 * ipsv[si][2]^j2 * zk[si])
                        va[si] += z
                        vb[si] += z
                    end
                end
            end
            M = [R_sampled(0) for i = 1:ts, j = 1:ts]
            ks = collect(keys(acc))
            ents = Vector{Any}(undef, length(ks))
            Threads.@threads for q in eachindex(ks)
                v = acc[ks[q]]
                ents[q] = SampledMPolyRingElem(R_sampled, [toQQ(x) for x in v])
            end
            for q in eachindex(ks)
                M[ks[q]...] = ents[q]
            end
            matrices[lambda] = M
            acc = nothing; ents = nothing
            GC.gc()   # BigFloat temporaries are malloc-backed; collect per lambda to bound peak memory
        end
    end
    setprecision(BigFloat, oldprec)
    verbose && @info "Finished constructing constraint $k FAST ($(iterationtime)s)"
    tr = representatives[k+1]
    return Constraint(size(tr, 1) == 1 ? -1 : 0, matrices, Dict(), all_samples[k])
end

@eval LSC function las2_constraint(n, t, k, d_tensor, representatives, FF, all_samples,
        all_matrices, tuple_to_index, irreps, weven, blocksizes, verbose; sampled=true)
    if sampled
        return Main.las2_constraint_fast(n, t, k, d_tensor, representatives, FF, all_samples,
            all_matrices, tuple_to_index, irreps, weven, blocksizes, verbose)
    else
        return las2_constraint_slow(n, t, k, d_tensor, representatives, FF, all_samples,
            all_matrices, tuple_to_index, irreps, weven, blocksizes, verbose; sampled=false)
    end
end
