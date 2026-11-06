# Allocation-free version of fastconstraint.jl's las2_constraint_fast: the zonal polynomials are
# evaluated with in-place Arb arithmetic (512 bits) on per-task buffers, so the 192 threads do not
# contend on BigFloat allocation and garbage collection (the BigFloat version spent ~2 h on the
# 4-point constraint at (16,18) for about 1e10 operations). Results are rounded to the same dyadic
# grid (2^-300) by the same toQQ. Requires fastconstraint.jl (LSC, las2_constraint_slow, toQQ, ZPREC).
using LasserreSphericalCodes, Nemo, ClusteredLowRankSolver
const AB = ClusteredLowRankSolver.Arblib

const ZT2 = Dict{Any,Any}()
const ZL2 = ReentrantLock()

# terms of the zonal entry (lambda, max(w1,w2), min(w1,w2)): coefficients (Arb), S-exponents, and the
# halved exponents of the factors 2(1 +- u1), 2(1 +- u2) (see evaluate_zonal_matrix)
function zterms2(n, lambda, w1, w2)
    i = max(w1, w2); j = min(w1, w2)
    key = (n, Tuple(lambda), i, j)
    c = lock(ZL2) do
        get(ZT2, key, nothing)
    end
    c === nothing || return c
    fn = "cache/zonalstore/$n/pol-$(n)-2-" * strip(string(lambda)) * "-" * string(i) * "-" * string(j) * ".txt"
    isfile(fn) || error("missing zonal file $fn")
    row_sums = [LSC.countels(lambda, k, i) for k = 1:2]
    col_sums = [LSC.countels(lambda, k, j) for k = 1:2]
    coefs = AB.Arb[]; evs = NTuple{4,Int}[]; fhs = NTuple{4,Int}[]; maxe = 0
    for l in eachline(fn)
        els = split(l)
        ev = [parse(Int, x) for x in els[1:end-1]]
        cf = AB.Arb(parse(Rational{BigInt}, els[end]); prec=ZPREC)
        exp_mat = reshape(ev, 2, 2)
        cur_rows = [sum(exp_mat[k, :]) for k = 1:2]
        cur_cols = [sum(exp_mat[:, k]) for k = 1:2]
        fa = [row_sums[1] - cur_rows[1], row_sums[2] - cur_rows[2], col_sums[1] - cur_cols[1], col_sums[2] - cur_cols[2]]
        @assert all(iseven, fa) && all(>=(0), fa)
        push!(coefs, cf); push!(evs, Tuple(ev)); push!(fhs, Tuple(div.(fa, 2)))
        maxe = max(maxe, maximum(ev), maximum(div.(fa, 2)))
    end
    val = (coefs, evs, fhs, maxe)
    lock(ZL2) do
        get!(ZT2, key, val)
    end
end

newarb() = AB.Arb(prec=ZPREC)

struct ZBuf
    u::Vector{AB.Arb}; S::Vector{AB.Arb}; P::Vector{AB.Arb}
    Spow::Matrix{AB.Arb}; Ppow::Matrix{AB.Arb}; t::AB.Arb
end
ZBuf(maxe) = ZBuf([newarb() for _ = 1:6], [newarb() for _ = 1:4], [newarb() for _ = 1:4],
                  [newarb() for _ = 1:4, _ = 1:maxe+1], [newarb() for _ = 1:4, _ = 1:maxe+1], newarb())

# res = zonal entry (w1, w2) at inner products u (u[1..6]), in place
function zeval2!(res::AB.Arb, terms, w1, w2, u::AbstractVector, B::ZBuf)
    coefs, evs, fhs, maxe = terms
    uu = B.u
    if max(w1, w2) != w1
        AB.set!(uu[1], u[2]); AB.set!(uu[2], u[1]); AB.set!(uu[3], u[3]); AB.set!(uu[4], u[5]); AB.set!(uu[5], u[4]); AB.set!(uu[6], u[6])
    else
        for q = 1:6; AB.set!(uu[q], u[q]); end
    end
    S = B.S; P = B.P; t = B.t
    # S = (u3+u4+u5+u6, u3+u4-u5-u6, u3-u4+u5-u6, u3-u4-u5+u6)
    AB.add!(t, uu[3], uu[4]); AB.add!(S[1], t, uu[5]); AB.add!(S[1], S[1], uu[6])
    AB.sub!(S[2], t, uu[5]); AB.sub!(S[2], S[2], uu[6])
    AB.sub!(t, uu[3], uu[4]); AB.add!(S[3], t, uu[5]); AB.sub!(S[3], S[3], uu[6])
    AB.sub!(S[4], t, uu[5]); AB.add!(S[4], S[4], uu[6])
    # P = (2(1+u1), 2(1-u1), 2(1+u2), 2(1-u2))
    AB.add!(P[1], uu[1], 1); AB.mul_2exp!(P[1], P[1], 1)
    AB.sub!(P[2], uu[1], 1); AB.neg!(P[2], P[2]); AB.mul_2exp!(P[2], P[2], 1)
    AB.add!(P[3], uu[2], 1); AB.mul_2exp!(P[3], P[3], 1)
    AB.sub!(P[4], uu[2], 1); AB.neg!(P[4], P[4]); AB.mul_2exp!(P[4], P[4], 1)
    Sp = B.Spow; Pp = B.Ppow
    for q = 1:4
        AB.one!(Sp[q, 1]); AB.one!(Pp[q, 1])
        for e = 1:maxe
            AB.mul!(Sp[q, e+1], Sp[q, e], S[q]); AB.mul!(Pp[q, e+1], Pp[q, e], P[q])
        end
    end
    AB.zero!(res)
    @inbounds for k in eachindex(coefs)
        ev = evs[k]; fh = fhs[k]
        AB.set!(t, coefs[k])
        for q = 1:4
            ev[q] > 0 && AB.mul!(t, t, Sp[q, ev[q]+1])
            fh[q] > 0 && AB.mul!(t, t, Pp[q, fh[q]+1])
        end
        AB.add!(res, res, t)
    end
    return res
end

function chunks_of(n)
    nt = Threads.nthreads()
    collect(Iterators.partition(1:n, max(1, cld(n, 4 * nt))))
end

function las2_constraint_fast2(n, t, k, d_tensor, representatives, FF, all_samples,
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
        # parse all needed zonal entries in parallel
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
            zterms2(n, lam, i, j)
        end
        maxe_all = maximum([zterms2(n, lam, i, j)[4] for (lam, i, j) in neededv]; init=0)
        # Gram entries of the samples as Arb
        G = [[AB.Arb(Rational{BigInt}(x); prec=ZPREC) for x in s] for s in samples]
        pairs = []
        for J1_index in 1:length(subs), J2_index in J1_index:length(subs)
            J1 = subs[J1_index]; J2 = subs[J2_index]
            sort(union(J1, J2)) == collect(1:k) || continue
            i1 = length(J1); i2 = length(J2)
            # the 6 inner products (u1, u2, u3..u6) per sample, as in las2_constraint_fast
            ipsv = [[newarb() for _ = 1:6] for _ = 1:ns]
            Threads.@threads for si = 1:ns
                v = ipsv[si]
                for q = 1:6; AB.one!(v[q]); end
                function ip(a, b)   # Gram entry of the sample (points a, b; 1 on the diagonal)
                    a == b && return nothing
                    idx = 0
                    for x = 1:k, y = x+1:k
                        idx += 1
                        ((x, y) == (min(a, b), max(a, b))) && return G[si][idx]
                    end
                end
                setip!(dst, a, b) = (g = ip(a, b); g === nothing ? AB.one!(dst) : AB.set!(dst, g))
                if i1 > 0 && i2 > 0
                    setip!(v[1], J1[min(1, i1)], J1[min(2, i1)]); setip!(v[2], J2[min(1, i2)], J2[min(2, i2)])
                    setip!(v[3], J1[min(1, i1)], J2[min(1, i2)]); setip!(v[4], J1[min(1, i1)], J2[min(2, i2)])
                    setip!(v[5], J1[min(2, i1)], J2[min(1, i2)]); setip!(v[6], J1[min(2, i1)], J2[min(2, i2)])
                elseif i1 > 0
                    setip!(v[1], J1[min(1, i1)], J1[min(2, i1)])
                elseif i2 > 0
                    setip!(v[2], J2[min(1, i2)], J2[min(2, i2)])
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
            acc = Dict{Tuple{Int,Int},Vector{AB.Arb}}()
            for (J1, J2, i1, i2, ipsv) in pairs
                adm1 = LSC.admissible_tuples(lambda, i1, weven[lambdaindex], d_tensor[lambdaindex])
                adm2 = LSC.admissible_tuples(lambda, i2, weven[lambdaindex], d_tensor[lambdaindex])
                (isempty(adm1) || isempty(adm2)) && continue
                kpairs = unique([(k1, k2) for (_, k1) in adm1 for (_, k2) in adm2])
                termsl = [zterms2(n, lambda, k1, k2) for (k1, k2) in kpairs]
                valsl = [[newarb() for _ = 1:ns] for _ in kpairs]
                Threads.@threads for ch in chunks_of(ns)
                    B = ZBuf(maxe_all)
                    for si in ch, q in eachindex(kpairs)
                        k1, k2 = kpairs[q]
                        zeval2!(valsl[q][si], termsl[q], k1, k2, ipsv[si], B)
                    end
                end
                qidx = Dict(kpairs[q] => q for q in eachindex(kpairs))
                f = J1 != J2 ? 1 : 1 // 2
                combos = Tuple{Int,Int,Int,Int,Int}[]   # (Za, Zb, j1, j2, q)
                for (a, (j1, k1)) in enumerate(adm1), (b, (j2, k2)) in enumerate(adm2)
                    Za = tuple_to_index[lambdaindex][(i1, j1, k1)]
                    Zb = tuple_to_index[lambdaindex][(i2, j2, k2)]
                    get!(() -> [newarb() for _ = 1:ns], acc, (Za, Zb))
                    get!(() -> [newarb() for _ = 1:ns], acc, (Zb, Za))
                    push!(combos, (Za, Zb, j1, j2, qidx[(k1, k2)]))
                end
                maxj = maximum(max(c[3], c[4]) for c in combos)
                accv = Dict(key => acc[key] for key in keys(acc))
                Threads.@threads for ch in chunks_of(ns)
                    u1p = [newarb() for _ = 0:maxj]; u2p = [newarb() for _ = 0:maxj]; z = newarb()
                    for si in ch
                        AB.one!(u1p[1]); AB.one!(u2p[1])
                        for e = 1:maxj
                            AB.mul!(u1p[e+1], u1p[e], ipsv[si][1]); AB.mul!(u2p[e+1], u2p[e], ipsv[si][2])
                        end
                        for (Za, Zb, j1, j2, q) in combos
                            AB.mul!(z, u1p[j1+1], u2p[j2+1])
                            AB.mul!(z, z, valsl[q][si])
                            f == 1 || AB.mul_2exp!(z, z, -1)
                            AB.add!(accv[(Za, Zb)][si], accv[(Za, Zb)][si], z)
                            AB.add!(accv[(Zb, Za)][si], accv[(Zb, Za)][si], z)
                        end
                    end
                end
            end
            M = [R_sampled(0) for i = 1:ts, j = 1:ts]
            ks = collect(keys(acc))
            ents = Vector{Any}(undef, length(ks))
            Threads.@threads for q in eachindex(ks)
                v = acc[ks[q]]
                ents[q] = SampledMPolyRingElem(R_sampled, [toQQ(BigFloat(x)) for x in v])
            end
            for q in eachindex(ks)
                M[ks[q]...] = ents[q]
            end
            matrices[lambda] = M
        end
    end
    setprecision(BigFloat, oldprec)
    verbose && @info "Finished constructing constraint $k FAST2 ($(iterationtime)s)"
    tr = representatives[k+1]
    return Constraint(size(tr, 1) == 1 ? -1 : 0, matrices, Dict(), all_samples[k])
end

@eval LSC function las2_constraint(n, t, k, d_tensor, representatives, FF, all_samples,
        all_matrices, tuple_to_index, irreps, weven, blocksizes, verbose; sampled=true)
    if sampled
        return Main.las2_constraint_fast2(n, t, k, d_tensor, representatives, FF, all_samples,
            all_matrices, tuple_to_index, irreps, weven, blocksizes, verbose)
    else
        return las2_constraint_slow(n, t, k, d_tensor, representatives, FF, all_samples,
            all_matrices, tuple_to_index, irreps, weven, blocksizes, verbose; sampled=false)
    end
end
