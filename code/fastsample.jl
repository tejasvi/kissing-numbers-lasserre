# Threaded LasserreSphericalCodes.sample_sos: sampling the 4-point SOS part (exact evaluation of every entry of the
# low-rank vectors at every sample, Rational{BigInt}) ran single-threaded and took > 14 h at delta = 22 (17358
# samples) on a test machine, 2026-10-21. Evaluating the Nemo polynomials themselves from many threads segfaulted (GC
# finalizers during the Nemo/GMP evaluation), so each entry is first converted once, single-threaded, to plain Julia
# (Rational{BigInt} coefficients, exponent vectors); the threaded part (over samples) uses only Julia arithmetic
# with a per-sample table of powers. The values are the exact rationals of the stock code.
using LasserreSphericalCodes, ClusteredLowRankSolver, Nemo

function k5_terms(p)
    p isa Union{Nemo.QQMPolyRingElem, Nemo.MPolyRingElem} || return nothing
    cs = Rational{BigInt}[Rational{BigInt}(c) for c in Nemo.coefficients(p)]
    es = Vector{Int}[collect(e) for e in Nemo.exponent_vectors(p)]
    cs, es
end

@eval LasserreSphericalCodes function sample_sos(v::LowRankMatPol, samples::Vector{Vector{T}}, FF) where T
    R = SampledMPolyRing(FF, samples)
    ents = [(i, j) for i in eachindex(v.vs) for j in eachindex(v.vs[i])]
    terms = [Main.k5_terms(v.vs[i][j]) for (i, j) in ents]
    if any(isnothing, terms) || FF !== Rational{BigInt}     # unexpected entry types: stock evaluation
        return LowRankMatPol(R.(v.lambda), [[R(v.vs[i][j]) for j in eachindex(v.vs[i])] for i in eachindex(v.vs)])
    end
    nv = length(first(samples))
    maxdeg = maximum(maximum(maximum(e) for e in t[2]; init=0) for t in terms; init=0)
    ns = length(samples)
    vals = [Vector{Rational{BigInt}}(undef, ns) for _ in ents]
    Threads.@threads :dynamic for s in 1:ns
        x = samples[s]
        pw = [Vector{Rational{BigInt}}(undef, maxdeg + 1) for _ = 1:nv]
        for a = 1:nv
            pw[a][1] = one(Rational{BigInt})
            for d = 1:maxdeg; pw[a][d+1] = pw[a][d] * Rational{BigInt}(x[a]); end
        end
        for (k, (cs, es)) in enumerate(terms)
            acc = zero(Rational{BigInt})
            for (c, e) in zip(cs, es)
                t = c
                for a = 1:nv
                    e[a] > 0 && (t *= pw[a][e[a]+1])
                end
                acc += t
            end
            vals[k][s] = acc
        end
    end
    out = [Vector{Any}(undef, length(v.vs[i])) for i in eachindex(v.vs)]
    for (k, (i, j)) in enumerate(ents)
        out[i][j] = SampledMPolyRingElem(R, vals[k])
    end
    LowRankMatPol(R.(v.lambda), [[x for x in o] for o in out])
end
