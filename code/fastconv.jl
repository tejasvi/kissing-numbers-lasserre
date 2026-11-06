# Faster ClusteredLowRankSDP(problem) conversion for ClusteredLowRankSolver 1.0.3. The original fills the
# sampled constraint matrices serially, and every entry evaluation of a SampledMPolyRingElem binary-searches
# the parent ring's sorted sample list, comparing Vector{Rational{BigInt}} with Vector{QQFieldElem} (BigInt
# conversions on every comparison): ~1500 s at (14,16), ~9500 s at (16,18). Here the sample-index maps are
# computed once per (ring, constraint), the evaluations run threaded over samples, and the results are
# inserted in the original order, so the resulting SDP is identical.
using ClusteredLowRankSolver
const CV = ClusteredLowRankSolver
const CVA = CV.Arblib
const ArbRefMatrix = CVA.ArbRefMatrix

canon_sample(s::AbstractVector) = Tuple(Rational{BigInt}(x) for x in s)
canon_sample(s) = (Rational{BigInt}(s),)

# collect the parent rings of all SampledMPolyRingElem inside a block coefficient
function collect_parents!(acc::IdDict, x)
    if x isa CV.SampledMPolyRingElem
        acc[CV.parent(x)] = true
    elseif x isa CV.LowRankMatPol
        for y in x.lambda; collect_parents!(acc, y); end
        for w in x.vs, y in w; collect_parents!(acc, y); end
        for w in x.ws, y in w; collect_parents!(acc, y); end
    elseif x isa AbstractMatrix
        for y in x; collect_parents!(acc, y); end
    end
    acc
end

struct FastEv
    maps::IdDict{Any,Vector{Int}}   # parent ring => index in its samples for each constraint sample
    prec::Int
end

function arb_of(q, prec)
    if q isa CV.Nemo.QQFieldElem || q isa Rational
        r = Rational{BigInt}(q)
        a = CVA.Arb(numerator(r); prec=prec)
        denominator(r) == 1 || CVA.div!(a, a, CVA.Arb(denominator(r); prec=prec))
        return a
    end
    return CVA.Arb(q, prec=prec)
end

function fev(F::FastEv, x, idx, sample)
    if x isa CV.SampledMPolyRingElem
        m = get(F.maps, CV.parent(x), nothing)
        m === nothing && return CV.sampleevaluate(x, sample, prec=F.prec)
        return arb_of(x.evaluations[m[idx]], F.prec)
    end
    return CV.sampleevaluate(x, sample, prec=F.prec)
end

function fev_block(F::FastEv, p, idx, sample, scaling)
    prec = F.prec
    if p isa CV.LowRankMatPol
        return CV.LowRankMat([arb_of(scaling, prec) * fev(F, v, idx, sample) for v in p.lambda],
                   [ArbRefMatrix([fev(F, v, idx, sample) for v in w], prec=prec) for w in p.ws],
                   [ArbRefMatrix([fev(F, v, idx, sample) for v in w], prec=prec) for w in p.vs])
    elseif p isa Matrix
        res = ArbRefMatrix(size(p)..., prec=prec)
        for i in eachindex(p)
            res[i] = fev(F, p[i], idx, sample)
        end
        return scaling == 1 ? res : CVA.Arb(scaling, prec=prec) * res
    else
        return CV.sampleevaluate(p, sample, scaling=scaling, prec=prec)
    end
end

function fastconv_fill!(v, sos, constraintindices, subblock_keys_dict, highranks, nsubblocks, subblocksizes, prec)
    jobs = Tuple{Int,Int,Int}[]
    i = 1
    for ci in constraintindices, idx in eachindex(sos.constraints[ci].samples)
        push!(jobs, (i, ci, idx)); i += 1
    end
    # index maps per constraint
    Fs = Dict{Int,FastEv}()
    for ci in constraintindices
        c = sos.constraints[ci]
        parents = IdDict()
        for blk in keys(c.matrixcoeff)
            collect_parents!(parents, c.matrixcoeff[blk])
        end
        maps = IdDict{Any,Vector{Int}}()
        cs = [canon_sample(s) for s in c.samples]
        for R in keys(parents)
            d = Dict(canon_sample(s) => k for (k, s) in enumerate(R.samples))
            mp = Int[]
            ok = true
            for key in cs
                k = get(d, key, 0)
                k == 0 && (ok = false; break)
                push!(mp, k)
            end
            ok && (maps[R] = mp)
        end
        Fs[ci] = FastEv(maps, prec)
    end
    res = Vector{Any}(undef, length(jobs))
    Threads.@threads :dynamic for q in eachindex(jobs)
        ii, ci, idx = jobs[q]
        c = sos.constraints[ci]
        sample = c.samples[idx]
        out = Tuple{Int,Int,Int,Any}[]
        for blk in keys(c.matrixcoeff)
            r, s = CV.subblock(blk)
            push!(out, (subblock_keys_dict[CV.name(blk)], r, s, fev_block(Fs[ci], c.matrixcoeff[blk], idx, sample, c.scalings[idx])))
        end
        res[q] = out
    end
    for q in eachindex(jobs)
        ii = jobs[q][1]
        for (vidx, r, s, val) in res[q]
            v[vidx][r, s][ii] = val
        end
        for l in keys(highranks)
            highranks[l] || continue
            nsubblocks[l] == 1 && continue   # identical to the original's hvcat of a single block
            vidx = subblock_keys_dict[l]
            all_blocks = ArbRefMatrix[get(v[vidx][r, s], ii, ArbRefMatrix(zeros(subblocksizes[l], subblocksizes[l]), prec=prec)) for r = 1:nsubblocks[l] for s = 1:nsubblocks[l]]
            m = hvcat(nsubblocks[l], all_blocks...)
            if !iszero(m)
                v[vidx][1, 1][ii] = m
            end
        end
        res[q] = nothing
    end
    return
end

let src = read(joinpath(pkgdir(CV), "src", "interface.jl"), String)
    i0 = findfirst("function ClusteredLowRankSDP(sos::Problem;", src)
    i1 = findnext("\n\"\"\"\n    convert_to_prec(sdp", src, first(i0))
    code = src[first(i0):first(i1)-1]
    a = findfirst("        i = 1\n        #This can take quite long", code)
    b = findfirst("        #contract high-rank blocks", code)
    @assert a !== nothing && b !== nothing "fastconv anchors not found"
    code = code[1:first(a)-1] *
        "        Main.fastconv_fill!(v, sos, constraintindices, subblock_keys_dict, highranks, nsubblocks, subblocksizes, prec)\n" *
        code[first(b):end]
    Base.eval(CV, Meta.parse(code))
end
