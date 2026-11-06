# Rounding speedups for roundsol.jl (ClusteredLowRankSolver 1.0.3 exact_solution), for the no-kernel recipe
# (kernel_use_primal=false, kernel_errbound tiny):
#  - kernel detection without the primal runs a BigFloat SVD of every dual block (GenericLinearAlgebra,
#    single-threaded), > 1 h for (14,20). A block whose smallest eigenvalue exceeds kernel_errbound has no kernel
#    vectors by the SVD's own criterion (it counts singular values below kernel_errbound), and a Cholesky
#    factorization of Y - kernel_errbound * I shows that much faster. k5_prescreen! tests all blocks in parallel;
#    basis_transformations then runs the SVD only for the blocks that fail.
#  - with no kernel vectors the basis transformation of a block is the identity; transform() then returns the
#    block unchanged instead of multiplying every polynomial vector by an identity matrix.
#  - the projection works on an 11172 x ~1.2e5 rational system at (14,20): the row-wise denominator clearing
#    (preprocess_rows!), the QQ -> ZZ conversion and the ZZ -> GF(p) conversion before the modular RREF were Julia
#    loops over ~1e9 entries each; here they are the FLINT routines fmpq_mat_get_fmpz_mat_rowwise,
#    fmpq_mat_get_fmpz_mat and fmpz_mat_get_nmod_mat (same results, checked against the stock code).
using ClusteredLowRankSolver, LinearAlgebra
if !isdefined(@__MODULE__, :K5_NOKERNEL)
const CLRS4 = ClusteredLowRankSolver
const K5_NOKERNEL = Dict{Any,Bool}()

function k5_prescreen!(sol, settings)
    empty!(K5_NOKERNEL)
    ks = collect(keys(sol.matrixvars))
    sort!(ks, by=k -> -size(sol.matrixvars[k], 1))
    res = Vector{Bool}(undef, length(ks))
    t = @elapsed Threads.@threads :dynamic for i in eachindex(ks)
        m = BigFloat.(CLRS4.matrixvar(sol, ks[i]))
        res[i] = size(m, 1) == 0 || isposdef(Symmetric(m - BigFloat(settings.kernel_errbound) * I))
    end
    for i in eachindex(ks)
        K5_NOKERNEL[ks[i]] = res[i]
    end
    println("  prescreen: $(count(res)) of $(length(res)) blocks have lmin > $(settings.kernel_errbound) ($(round(t, digits=1))s)")
    for i in eachindex(ks)
        res[i] || println("    block $(ks[i]) (size $(size(sol.matrixvars[ks[i]], 1))) goes to the SVD")
    end
    flush(stdout)
end

k5_isidentity(B) = size(B, 1) == size(B, 2) && all(B[i, j] == (i == j ? 1 : 0) for i in axes(B, 1), j in axes(B, 2))

let src = read(joinpath(pkgdir(CLRS4), "src", "rounding.jl"), String)
    i = findfirst("function basis_transformations(primalsol::PrimalSolution, sol::DualSolution;", src)
    k = findnext("\nend\n", src, last(i))
    code = src[first(i):first(k)+4]
    old = "list = detecteigenvectors(pm, m; settings=settings, FF=FF, g=g)"
    @assert count(old, code) == 1 "fastround: detecteigenvectors anchor"
    code = replace(code, old => "list = (!settings.kernel_use_primal && get(Main.K5_NOKERNEL, k, false)) ? [] : detecteigenvectors(pm, m; settings=settings, FF=FF, g=g)")
    Base.eval(CLRS4, Meta.parse(code))
end

@eval CLRS4 function transform(p::LowRankMatPol, Binv, s)
    s == 0 && Main.k5_isidentity(Binv) && return LowRankMatPol(p.lambda, copy.(p.vs), copy.(p.ws))
    vs = [(Binv*v)[s+1:end] for v in p.vs]
    ws = [(Binv*v)[s+1:end] for v in p.ws]
    LowRankMatPol(p.lambda, vs, ws)
end
const K5FL = CLRS4.Nemo.libflint
function k5_qq(Z::CLRS4.ZZMatrix)
    Q = CLRS4.zero_matrix(CLRS4.QQ, CLRS4.nrows(Z), CLRS4.ncols(Z))
    ccall((:fmpq_mat_set_fmpz_mat, K5FL), Nothing, (Ref{CLRS4.QQMatrix}, Ref{CLRS4.ZZMatrix}), Q, Z)
    Q
end
function k5_zz(A::CLRS4.QQMatrix)     # A must be integral
    Z = CLRS4.zero_matrix(CLRS4.ZZ, CLRS4.nrows(A), CLRS4.ncols(A))
    ok = ccall((:fmpq_mat_get_fmpz_mat, K5FL), Cint, (Ref{CLRS4.ZZMatrix}, Ref{CLRS4.QQMatrix}), Z, A)
    ok == 1 || error("k5_zz: matrix is not integral")
    Z
end
# rows of A times the lcm of their denominators, and those lcms
function k5_rowwise(A::CLRS4.QQMatrix)
    r = CLRS4.nrows(A)
    Z = CLRS4.zero_matrix(CLRS4.ZZ, r, CLRS4.ncols(A))
    den = ccall((:_fmpz_vec_init, K5FL), Ptr{Int}, (Int,), max(r, 1))
    ccall((:fmpq_mat_get_fmpz_mat_rowwise, K5FL), Nothing, (Ref{CLRS4.ZZMatrix}, Ptr{Int}, Ref{CLRS4.QQMatrix}), Z, den, A)
    d = [CLRS4.ZZRingElem() for _ = 1:r]
    for i = 1:r
        ccall((:fmpz_set, K5FL), Nothing, (Ref{CLRS4.ZZRingElem}, Ptr{Int}), d[i], den + 8 * (i - 1))
    end
    ccall((:_fmpz_vec_clear, K5FL), Nothing, (Ptr{Int}, Int), den, max(r, 1))
    Z, d
end
@eval CLRS4 function preprocess_rows!(A, b; include_b=false)
    if include_b
        Z, _ = Main.k5_rowwise(hcat(A, b))
        nc = ncols(A)
        return Main.k5_qq(Z[:, 1:nc]), Main.k5_qq(Z[:, nc+1:nc+1])
    end
    Z, d = Main.k5_rowwise(A)
    b2 = deepcopy(b)
    for r in 1:nrows(b2)
        b2[r, 1] = b2[r, 1] * d[r]
    end
    return Main.k5_qq(Z), b2
end
# size of the random primes for the modular RREF (bits, env K5_PBITS, default 25: small enough for FLINT's fast
# floating-point matrix kernels; a 62-bit prime made the RREF of 11172 x 255452 take > 1 h instead of 17 min)
const K5_PBITS = Ref(parse(Int, get(ENV, "K5_PBITS", "25")))
# pivots of [A b]; if b comes out as a pivot (inconsistent), repeat with a second independent random prime and use that
# answer: a false inconsistency needs the prime to divide a row's scaling, which two random primes almost never both do
function k5_pivots_twice(Ab)
    Z = k5_zz(Ab)
    piv = CLRS4.find_pivots_modular(Z)
    if size(Ab, 2) == piv[end]
        piv2 = CLRS4.find_pivots_modular(Z)
        println("  (inconsistent mod the first prime; second prime: $(size(Ab, 2) == piv2[end] ? "inconsistent too" : "consistent"))"); flush(stdout)
        return piv2
    end
    return piv
end
# the stock starting prime min(maximum(abs.(A)), 10^4) is a Julia pass over all entries; for big matrices (whose
# integral entries exceed 10^4 anyway) start at 10^4 directly
@eval CLRS4 function find_pivots_modular(A::ZZMatrix; maxprimes = 3)
    # a random prime near 2^62 instead of next_prime(10^4): the rows are scaled by the lcm of their denominators,
    # including the denominator of the right-hand side, which comes from the 40-digit rational approximation of the
    # numerical solution and can contain any small prime; a row whose matrix entries all become divisible by p while
    # its right-hand side does not is then a false inconsistency, the same one in every attempt (at (14,20): rank of
    # [A b] mod 10007 = 11172 with 255452 columns, i.e. a single missing direction, six attempts in a row)
    p = ZZ(rand(UInt64(2)^(Main.K5_PBITS[] - 1):UInt64(2)^Main.K5_PBITS[]))
    numps = 1
    history = []
    for numps = 1:maxprimes
        p = next_prime(p)
        pivots = find_pivots_modular(A, p)
        if length(pivots) == nrows(A)
            if numps > 1
                println("  Needed to do the RREF for $numps primes")
                println("  Succeeded with p = $p")
            end
            return pivots
        elseif numps >= maxprimes
            push!(history, pivots)
            bestlength = maximum(unique(length.(history)))
            pivots = history[findfirst(x->length(x)==bestlength, history)]
            return pivots
        end
        push!(history, pivots)
    end
    return nothing
end
@eval CLRS4 function find_pivots_modular(A::ZZMatrix, p)
    F = Nemo.Native.GF(Int(p))
    M = Nemo.fpMatrix(UInt(p), A)
    M.base_ring = F
    rnk, res = Nemo.rref(M)
    pivots = Int[]
    z = F(0)
    for i=1:nrows(M)
        startcol = length(pivots) > 0 ? last(pivots)+1 : i
        for j=startcol:ncols(res)
            if !(res[i,j] == z)
                push!(pivots, j)
                break
            end
        end
    end
    return unique(pivots)
end
let src = read(joinpath(pkgdir(CLRS4), "src", "rounding.jl"), String)
    i = findfirst("function project_to_affine_space(A::QQMatrix, b::QQMatrix;", src)
    k = findnext("\nend\n", src, last(i))
    code = src[first(i):first(k)+4]
    inc_old = "        println(\"  The system is inconsistent; taking more columns\")\n"
    @assert count(inc_old, code) == 1 "fastround: inconsistency anchor"
    code = replace(code, inc_old => "        println(\"  The system is inconsistent; taking more columns (rank of [A b] mod p: \$(length(pivots)) for \$(nrows(Ab)) rows and \$(ncols(Ab) - 1) columns)\")\n")
    for (old, new) in ("find_pivots_modular(ZZ.(Ab))" => "Main.k5_pivots_twice(Ab)",
                       "find_pivots_modular(transpose(ZZ.(A[:, pivots])))" => "find_pivots_modular(transpose(Main.k5_zz(A[:, pivots])))",
                       "find_pivots_modular(ZZ.(A[rows, nonpivots]))" => "find_pivots_modular(Main.k5_zz(A[rows, nonpivots]))")
        @assert count(old, code) == 1 "fastround: projection anchor $old"
        code = replace(code, old => new)
    end
    Base.eval(CLRS4, Meta.parse(code))
end

# pseudo-inverse solve: the stock code scales the columns and rows of A*A^T by their gcds with two DENSE products
# by diagonal matrices (each as costly as A*A^T itself at 11172 x 11172); here the scalings are entrywise
# (fmpz_gcd / fmpq_div_fmpz on the matrix entries), the rest as in the stock code
_k5_entry(M, i, j) = ccall((:fmpq_mat_entry, K5FL), Ptr{Nothing}, (Ref{CLRS4.QQMatrix}, Int, Int), M, i - 1, j - 1)
function k5_gcds(M::CLRS4.QQMatrix, cols::Bool)
    r, c = CLRS4.nrows(M), CLRS4.ncols(M)
    g = [CLRS4.ZZRingElem(0) for _ = 1:(cols ? c : r)]
    for j = 1:c, i = 1:r
        k = cols ? j : i
        ccall((:fmpz_gcd, K5FL), Nothing, (Ref{CLRS4.ZZRingElem}, Ref{CLRS4.ZZRingElem}, Ptr{Nothing}), g[k], g[k], _k5_entry(M, i, j))
    end
    g
end
function k5_divide!(M::CLRS4.QQMatrix, g, cols::Bool)
    r, c = CLRS4.nrows(M), CLRS4.ncols(M)
    for j = 1:c, i = 1:r
        e = _k5_entry(M, i, j)
        ccall((:fmpq_div_fmpz, K5FL), Nothing, (Ptr{Nothing}, Ptr{Nothing}, Ref{CLRS4.ZZRingElem}), e, e, g[cols ? j : i])
    end
    M
end
@eval CLRS4 function solve_system_pseudoinverse(A,b)
    At = transpose(A)
    if ncols(A) > nrows(A)
        AAt = A*At
        dr = Main.k5_gcds(AAt, true); Main.k5_divide!(AAt, dr, true)     # AAt * Dr, Dr = diag(1/dr)
        dl = Main.k5_gcds(AAt, false); Main.k5_divide!(AAt, dl, false)   # Dl * AAt, Dl = diag(1/dl)
        newb = deepcopy(b)
        for i = 1:nrows(newb); newb[i, 1] = newb[i, 1] // dl[i]; end
        y = Nemo._solve_dixon(AAt, newb)
        for i = 1:nrows(y); y[i, 1] = y[i, 1] // dr[i]; end
        v = At * y
    else
        AtA = At * A
        Dr = matrix(QQ, Diagonal([QQ(1)//gcd(Vector(numerator.(AtA[:,k]))) for k in 1:size(AtA,2)]))
        AtA = AtA * Dr
        Dl = matrix(QQ, Diagonal([QQ(1)//gcd(Vector(numerator.(AtA[k,:]))) for k in 1:size(AtA,1)]))
        AtA = Dl * AtA
        newb = Dl * (At * b)
        y = Nemo._solve_dixon(AtA, newb)
        v = Dr * y
    end
    return v
end

# column selection for the projection: the stock code keeps the diagonal and superdiagonal of every block plus
# random entries (redundancyfactor x rows in total). At (14,20) that column subset was inconsistent twice (10x, 12x):
# random entries of the large 4-point SOS blocks cover few monomials each. Here every entry of the kernel blocks and
# of the 1-, 2- and 3-point SOS blocks (30476 entries at (14,20); their polynomials are dense) is kept, the 4-point
# SOS blocks contribute their diagonal and superdiagonal, and random entries fill up to the same total as the stock rule.
k5_bigsos(k) = k isa Tuple && first(k) == :sos4
@eval CLRS4 function select_columns(problem::Problem, sol::DualSolution, redundancyfactor=6; verbose=true)
    nconstraints = sum(x->length(x.samples), problem.constraints)
    x = vectorize(sol)
    nvars = length(x)
    if redundancyfactor < 0
        return collect(1:nvars)
    end
    v = as_dual_solution(sol, zeros(Int,nvars))
    for (k, m) in v.matrixvars
        m = zero(m)
        if Main.k5_bigsos(k)
            for i in axes(m,1)
                m[i,i] = 1
                if size(m,2) >= i+1
                    m[i,i+1] = m[i+1,i] = 1
                end
            end
        else
            m .= 1
        end
        v.matrixvars[k] = m
    end
    for (k, f) in v.freevars
        v.freevars[k] = 1
    end
    vvec = vectorize(v)
    pivot_cols = [i for (i,val) in enumerate(vvec) if val >= 1]
    nfixed = length(pivot_cols)
    nneeded = redundancyfactor * nconstraints - length(pivot_cols)
    nneeded = max((redundancyfactor-2) * nconstraints, nneeded)
    notchosen = shuffle!([i for (i,val) in enumerate(vvec) if val == 0])
    append!(pivot_cols, notchosen[1:min(nneeded, length(notchosen))])
    verbose && println("  Reducing the system from $(length(vvec)) columns to $(length(pivot_cols)) columns ($nfixed kept by block type)")
    return pivot_cols
end

# retries after an inconsistent column subset: (1) the residual polynomials -slacks(problem, sol) of the rounded
# solution are the same in every attempt (same rounded x) and took ~80 min of the ~85-min system construction at
# (14,20); they are kept from the previous attempt (keyed by the problem and a hash of x); (2) the redundancy grows by
# K5_RS_REDUND_STEP per attempt (stock: 2).
const K5_SLACKS = Dict{Any,Any}()
function k5_slacks(problem, sol)
    key = (objectid(problem), hash(CLRS4.vectorize(sol)))
    if haskey(K5_SLACKS, key)
        println("  (residual polynomials reused from the previous attempt)"); flush(stdout)
        return K5_SLACKS[key]
    end
    r = -CLRS4.slacks(problem, sol)
    empty!(K5_SLACKS)
    K5_SLACKS[key] = r
    return r
end
let src = read(joinpath(pkgdir(CLRS4), "src", "interface.jl"), String)
    i = findfirst("function partial_linearsystem(problem::Problem, sol::DualSolution, columns::DualSolution;", src)
    k = findnext("\nend\n", src, last(i))
    code = src[first(i):first(k)+4]
    old = "rhs = -slacks(problem, sol) # b - Ax"
    @assert count(old, code) == 1 "fastround: slacks anchor"
    code = replace(code, old => "rhs = Main.k5_slacks(problem, sol) # b - Ax (kept across retries)")
    Base.eval(CLRS4, Meta.parse(code))
end

# Incremental column selection (K5_INCREMENTAL=1): the columns are taken in priority order -- every entry of the
# kernel blocks and the 1-3-point SOS blocks and the free variables, then the 4-point SOS entries by decreasing degree
# of their pair of basis polynomials (the top-degree monomials of the 4-point identity are reached only through
# those pairs) -- in batches of K5_INC_BATCH columns. After each batch the system of (pivot columns so far) + batch is
# reduced mod p and only its pivot columns are kept, so memory stays at 11172 x (pivots + batch). It stops as soon as
# the right-hand side is not a pivot, i.e. the kept columns span it mod p, and returns those columns plus up to
# K5_INC_EXTRA further columns of the last batch (for the pseudo-inverse).
function k5_basis_degrees(problem, k, s)
    deg = zeros(Int, s)
    for c in problem.constraints
        haskey(c.matrixcoeff, k) || continue
        m = c.matrixcoeff[k]
        if m isa CLRS4.LowRankMatPol
            for vecs in (m.vs, m.ws), v in vecs, a in 1:min(s, length(v))
                iszero(v[a]) || (deg[a] = max(deg[a], CLRS4.Nemo.total_degree(v[a])))
            end
        end
    end
    deg
end
function k5_incremental_columns(problem, sol, settings, monomial_bases; verbose=true)
    x = CLRS4.vectorize(sol)
    nvars = length(x)
    idx = CLRS4.as_dual_solution(sol, collect(1:nvars))
    first_cols = Int[]
    big = Tuple{Int,Float64,Int}[]
    for (k, m) in idx.matrixvars
        s = size(m, 1)
        if k5_bigsos(k)
            deg = k5_basis_degrees(problem, k, s)
            for i in 1:s, j in i:s
                push!(big, (deg[i] + deg[j], rand(), m[i, j]))
            end
        else
            for i in 1:s, j in i:s
                push!(first_cols, m[i, j])
            end
        end
    end
    append!(first_cols, collect(values(idx.freevars)))
    sort!(big, by=t -> (-t[1], t[2]))
    order = vcat(first_cols, [t[3] for t in big])
    verbose && println("  incremental selection: $(length(first_cols)) columns of small blocks, then $(length(big)) 4-point SOS entries (top pair degree $(isempty(big) ? 0 : big[1][1]))")
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
        verbose && println("  incremental selection: $(length(cand)) columns -> rank $(length(P)) of $(size(Ab, 1)) rows, b $(consistent ? "spanned" : "not spanned") ($(round(t, digits=1))s)"); flush(stdout)
        if consistent
            Pset = Set(P)
            extra = [c for c in newc if !(c in Pset)][1:min(nextra, end)]
            return vcat(P, extra)
        end
    end
    error("incremental selection: all columns used and the system is still inconsistent")
end
let src = read(joinpath(pkgdir(CLRS4), "src", "rounding.jl"), String)
    i = findfirst("function project_to_affine_space(problem::Problem, sol::DualSolution;", src)
    k = findnext("\nend\n", src, last(i))
    code = src[first(i):first(k)+4]
    old1 = "extra_redundancy += 2"
    @assert count(old1, code) == 1 "fastround: redundancy step anchor"
    code = replace(code, old1 => "extra_redundancy += parse(Int, get(ENV, \"K5_RS_REDUND_STEP\", \"2\"))")
    old2 = "column_selection = select_columns(problem, sol, settings.redundancyfactor + extra_redundancy, verbose=verbose)"
    @assert count(old2, code) == 1 "fastround: column selection anchor"
    code = replace(code, old2 => "column_selection = get(ENV, \"K5_INCREMENTAL\", \"0\") == \"1\" ? Main.k5_incremental_columns(problem, sol, settings, monomial_bases; verbose=verbose) : select_columns(problem, sol, settings.redundancyfactor + extra_redundancy, verbose=verbose)")
    Base.eval(CLRS4, Meta.parse(code))
end

@eval CLRS4 function transform(p::Matrix{T}, Binv, s; g=1) where T
    s == 0 && !(T <: Integer) && Main.k5_isidentity(Binv) && return copy(p)   # (Int matrices become QQ in the stock code)
    if T == BigFloat
        Binv = generic_embedding.(Binv, g, base_ring=BigFloat)
    end
    Binv[s+1:end, :]*p*transpose(Binv)[:, s+1:end]
end
end
