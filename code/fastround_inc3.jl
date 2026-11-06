# fastround_inc3.jl: hybrid + random incremental column selection (included after fastround.jl by roundsol_inc3.jl;
# replaces k5_incremental_columns). At n = 7 (14,20): the random selection (small blocks, 4-point diagonal and
# superdiagonal, random fill) was inconsistent seven times with [A b] of full rank, so at most one direction short;
# the hybrid order of fastround_inc2.jl (same first columns, then 4-point entries by decreasing pair degree) reached
# rank 11161 with its first batch but only 11169 of 11172 after four batches: the last directions need 4-point entries
# of lower pair degree, which random entries supply. Here: the same first columns, then the K5_INC_TOP highest-degree
# 4-point entries (default 45644, which fills the first batch exactly as in fastround_inc2.jl), then the remaining
# 4-point entries in RANDOM order, in batches of K5_INC_BATCH until the right-hand side is spanned mod p; plus
# K5_INC_EXTRA further columns for the pseudo-inverse.
using Random
function k5_incremental_columns(problem, sol, settings, monomial_bases; verbose=true)
    x = CLRS4.vectorize(sol)
    nvars = length(x)
    idx = CLRS4.as_dual_solution(sol, collect(1:nvars))
    first_cols = Int[]
    big = Tuple{Int,Float64,Int}[]
    nband = 0
    for (k, m) in idx.matrixvars
        s = size(m, 1)
        if k5_bigsos(k)
            deg = k5_basis_degrees(problem, k, s)
            for i in 1:s, j in i:s
                if j <= i + 1
                    push!(first_cols, m[i, j]); nband += 1
                else
                    push!(big, (deg[i] + deg[j], rand(), m[i, j]))
                end
            end
        else
            for i in 1:s, j in i:s
                push!(first_cols, m[i, j])
            end
        end
    end
    append!(first_cols, collect(values(idx.freevars)))
    sort!(big, by=t -> (-t[1], t[2]))
    ntop = min(parse(Int, get(ENV, "K5_INC_TOP", "45644")), length(big))
    rest = [t[3] for t in big[ntop+1:end]]
    shuffle!(rest)
    order = vcat(first_cols, [t[3] for t in big[1:ntop]], rest)
    verbose && println("  hybrid+random incremental selection: $(length(first_cols)) first columns (incl. $nband 4-point diagonal/superdiagonal entries), $ntop top-degree 4-point entries (pair degree $(isempty(big) ? 0 : big[1][1]) to $(ntop > 0 ? big[ntop][1] : 0)), then $(length(rest)) 4-point entries in random order")
    batch = parse(Int, get(ENV, "K5_INC_BATCH", "100000"))
    nextra = parse(Int, get(ENV, "K5_INC_EXTRA", "600"))
    xr = CLRS4.roundx(x, power=settings.approximation_decimals)
    solr = CLRS4.as_dual_solution(sol, xr)
    P = Int[]
    pos = 1
    while pos <= length(order)
        newc = order[pos:min(pos + batch - 1, length(order))]
        pos += batch
        cand = vcat(P, newc)
        t = @elapsed begin
            A, b = CLRS4.partial_linearsystem(problem, solr, cand; monomial_bases=monomial_bases)
            A, b = CLRS4.preprocess_rows!(A, b, include_b=true)
            Ab = hcat(A, b)
            piv = CLRS4.find_pivots_modular(k5_zz(Ab))
        end
        consistent = !(size(Ab, 2) in piv)
        P = cand[[q for q in piv if q <= length(cand)]]
        verbose && println("  hybrid+random incremental selection: $(length(cand)) columns -> rank $(length(P)) of $(size(Ab, 1)) rows, b $(consistent ? "spanned" : "not spanned") ($(round(t, digits=1))s)"); flush(stdout)
        if consistent
            Pset = Set(P)
            extra = [c for c in newc if !(c in Pset)][1:min(nextra, end)]
            return vcat(P, extra)
        end
    end
    error("hybrid+random incremental selection: all columns used and the system is still inconsistent")
end
