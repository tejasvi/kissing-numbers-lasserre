# Hash-based deduplication of the low-rank vectors in ClusteredLowRankSolver 1.0.3's
# precompute_matrices_bilinear_pairings (run at the start of every solvesdp call). The stock code compares
# every vector with every earlier one (unique_idx) and then, for each vector, searches the unique list
# again (findfirst): O(n^2) Arb-vector comparisons per block, ~30 min per call at (16,18) where blocks have
# ~2e4 vectors. Here vectors are bucketed by the hash of their Float64 roundings and compared exactly only
# within a bucket; the unique list and the pointers (position in the unique list of the first equal vector)
# are identical to the stock ones.
using ClusteredLowRankSolver
const FP = ClusteredLowRankSolver

function fast_unique_ptr(vecs)
    buckets = Dict{UInt,Vector{Int}}()      # hash -> positions in uniq
    uniq = Int[]
    ptr = Vector{Int}(undef, length(vecs))
    for (i, v) in enumerate(vecs)
        h = UInt(0)
        for k in eachindex(v)
            h = hash(Float64(v[k]), h)
        end
        cands = get!(() -> Int[], buckets, h)
        found = 0
        for u in cands
            if vecs[uniq[u]] == v
                found = u; break
            end
        end
        if found == 0
            push!(uniq, i)
            found = length(uniq)
            push!(cands, found)
        end
        ptr[i] = found
    end
    return uniq, ptr
end

let src = read(joinpath(pkgdir(FP), "src", "solver.jl"), String)
    i0 = findfirst("function precompute_matrices_bilinear_pairings(", src)
    i1 = findnext("\nend\n", src, first(i0))
    code = src[first(i0):last(i1)]
    for side in ("right", "left")
        a = "                    unique_$side = unique_idx($side)\n"
        @assert count(a, code) == 1 "fastprecomp anchor (unique_$side)"
        code = replace(code, a => "                    unique_$side, ptrvec_$side = Main.fast_unique_ptr($side)\n")
        r = Regex("pntr_$side = Dict\\(\\(s,p,rnk\\) => findfirst\\(x-> $side\\[x\\]==$side\\[i\\], unique_$side\\) for \\(i,\\(s,p,rnk\\)\\) in enumerate\\(")
        @assert occursin(r, code) "fastprecomp anchor (pntr_$side)"
        code = replace(code, r => "pntr_$side = Dict((s,p,rnk) => ptrvec_$side[i] for (i,(s,p,rnk)) in enumerate(")
    end
    Base.eval(FP, Meta.parse(code))
end
