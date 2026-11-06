# Per-phase profile of every solver iteration: wall time, average parallelism (process CPU time / wall time) and
# GC time of each timed phase of ClusteredLowRankSolver 1.0.3 (the stock @elapsed sites in solvesdp,
# compute_T_decomposition! and compute_search_direction!, plus the per-call setup and the GC.gc(false) at the top
# of each iteration), printed as one "TPHASE" line per iteration. Also prints the full nested error when solvesdp
# catches an exception (supersedes showerr.jl: both patch solvesdp, the last eval wins).
using ClusteredLowRankSolver, LinearAlgebra
include(joinpath(@__DIR__, "eigmin_jl.jl"))
const IP = ClusteredLowRankSolver

const CPROF = Dict{Symbol,Vector{Float64}}()     # name => [wall, cpu, gc, count, allocated GiB]
const CPROF_LOCK = ReentrantLock()
const CLK_TCK = 100
function _cpu()
    s = read("/proc/self/stat", String)
    f = split(s[findlast(')', s)+2:end])     # fields from 3 (state) on: utime = field 14, stime = field 15
    (parse(Int, f[12]) + parse(Int, f[13])) / CLK_TCK
end
function _cprof_add(name, t, c, g, a=0.0)
    lock(CPROF_LOCK) do
        v = get!(() -> zeros(5), CPROF, name)
        v[1] += t; v[2] += c; v[3] += g; v[4] += 1; v[5] += a
    end
end
macro celapsed(name, ex)
    quote
        local _c0 = Main._cpu()
        local _g0 = Base.gc_num().total_time
        local _a0 = Base.gc_bytes()
        local _t = @elapsed $(esc(ex))
        Main._cprof_add($(esc(name)), _t, Main._cpu() - _c0, (Base.gc_num().total_time - _g0) / 1e9, (Base.gc_bytes() - _a0) / 2^30)
        _t
    end
end
const CPROF_ORDER = [:gcfalse, :time_R, :time_inv, :time_decomp, :time_schur, :time_cholS, :time_LinvB, :time_Q,
    :time_cholQ, :time_res, :time_predictor_dir, :time_corrector_dir, :time_Z, :time_rhs_x, :time_sys, :time_dX,
    :time_dY, :time_alpha, :objectives]
function _cprof_print(iter)
    lock(CPROF_LOCK) do
        io = IOBuffer()
        print(io, "TPHASE it=$iter")
        names = vcat([k for k in CPROF_ORDER if haskey(CPROF, k)], sort([k for k in keys(CPROF) if !(k in CPROF_ORDER)]))
        for k in names
            w, c, g, n, a = CPROF[k]
            nm = replace(string(k), "time_" => "")
            print(io, " $nm=", round(w, digits=1), "s/x", w > 0.05 ? round(c / w, digits=1) : 0.0,
                  g > 0.05 ? "/gc" * string(round(g, digits=1)) : "", a > 0.5 ? "/" * string(round(a, digits=1)) * "G" : "",
                  n > 1 ? "/n$(Int(n))" : "")
        end
        println(String(take!(io))); flush(stdout)
        empty!(CPROF)
    end
end

# Adaptive centering (K5_ADAPT=1, default): at the top of every iteration the centrality of the kernel blocks
# (size <= 64) is measured as lambda_min(L^T Y L) / mu with X = L L^T (1 on the central path). The (16,16) step
# collapse came from one such block ([11,0]) drifting to 9e-4 while all others stayed above 0.15; here the
# iteration's centering parameters are raised when a block drifts: below K5_CENT_MID (0.2) beta >= 0.5, below
# K5_CENT_LO (0.05) beta >= 0.7 and gamma <= 0.6, below 0.01 beta >= 0.85 / gamma <= 0.45, below 0.002 beta >= 0.95
# / gamma <= 0.35 (beta_feasible raised alongside).
const K5_CENT_LO = parse(Float64, get(ENV, "K5_CENT_LO", "0.05"))
const K5_CENT_MID = parse(Float64, get(ENV, "K5_CENT_MID", "0.2"))
function k5_adapt(X, Y, mu, bi, g, bf, iter)
    get(ENV, "K5_ADAPT", "1") == "1" || return bi, g, bf
    prec = precision(mu)
    blocks = [(j, l) for j in eachindex(X.blocks) for l in eachindex(X.blocks[j].blocks) if size(X.blocks[j].blocks[l], 1) <= 64]
    isempty(blocks) && return bi, g, bf
    cs = fill(Inf, length(blocks))
    muf = Float64(mu)
    Threads.@threads for i in eachindex(blocks)
        j, l = blocks[i]
        Xb = X.blocks[j].blocks[l]; Yb = Y.blocks[j].blocks[l]; s = size(Xb, 1)
        L = IP.ArbRefMatrix(s, s, prec=prec)
        if IP.approx_cholesky!(L, Xb) == 0
            cs[i] = -1.0; continue
        end
        Lt = IP.ArbRefMatrix(s, s, prec=prec); IP.Arblib.transpose!(Lt, L)
        T = IP.ArbRefMatrix(s, s, prec=prec); IP.Arblib.approx_mul!(T, Lt, Yb, prec=prec)
        W = IP.ArbRefMatrix(s, s, prec=prec); IP.Arblib.approx_mul!(W, T, L, prec=prec)
        Wf = Float64.(W) ./ muf
        # LAPACK (OpenBLAS, built for 64 threads) aborts once called from too many threads ("too many memory
        # regions": killed several runs on 2026-10-21)
        cs[i] = Main.k5_eigmin(Wf)    # pure Julia (eigmin_jl.jl): no BLAS from solver threads
    end
    k = argmin(cs); c = cs[k]
    nm = isdefined(Main, :K5_BLOCKNAMES) ? Main.K5_BLOCKNAMES[blocks[k][1]][blocks[k][2]] : blocks[k]
    bi2, g2, bf2 = bi, g, bf
    if c < K5_CENT_LO / 25        # < 0.002: almost pure centering
        bi2 = max(bi, IP.Arblib.Arb(95//100, prec=prec)); g2 = min(g, IP.Arblib.Arb(35//100, prec=prec)); bf2 = max(bf, IP.Arblib.Arb(9//10, prec=prec))
    elseif c < K5_CENT_LO / 5     # < 0.01 ((16,18) block [11,0] kept drifting at beta 0.7 / gamma 0.6)
        bi2 = max(bi, IP.Arblib.Arb(85//100, prec=prec)); g2 = min(g, IP.Arblib.Arb(45//100, prec=prec)); bf2 = max(bf, IP.Arblib.Arb(7//10, prec=prec))
    elseif c < K5_CENT_LO
        bi2 = max(bi, IP.Arblib.Arb(7//10, prec=prec)); g2 = min(g, IP.Arblib.Arb(6//10, prec=prec)); bf2 = max(bf, IP.Arblib.Arb(5//10, prec=prec))
    elseif c < K5_CENT_MID
        bi2 = max(bi, IP.Arblib.Arb(5//10, prec=prec)); bf2 = max(bf, IP.Arblib.Arb(3//10, prec=prec))
    end
    println("CENT it=$iter min=", round(c, sigdigits=3), " at $nm (s=$(size(X.blocks[blocks[k][1]].blocks[blocks[k][2]], 1)))",
            " beta=", round(Float64(bi2), digits=2), " gamma=", round(Float64(g2), digits=2)); flush(stdout)
    return bi2, g2, bf2
end

# B <- X^-1 B in place, with X = L L^T given by L (X_inv in the solver): FLINT's solve_cho_precomp runs all columns in
# one thread; for blocks larger than 128 the columns are split into chunks solved in parallel tasks.
function k5_solve_cho!(B, L)
    s = size(B, 2)
    if s <= 128 || Threads.nthreads() == 1
        IP.Arblib.solve_cho_precomp!(B, L, B)
        return B
    end
    nch = min(Threads.nthreads(), cld(s, 16))
    bounds = [round(Int, i * s / nch) for i = 0:nch]
    Threads.@threads :dynamic for i = 1:nch
        c1 = bounds[i]; c2 = bounds[i+1]
        c2 > c1 || continue
        W = IP.ArbRefMatrix(0, 0, prec=precision(B))
        IP.Arblib.window_init!(W, B, 0, c1, size(B, 1), c2)
        T = IP.ArbRefMatrix(size(B, 1), c2 - c1, prec=precision(B))
        IP.Arblib.solve_cho_precomp!(T, L, W)
        IP.Arblib.set!(W, T)
        IP.Arblib.window_clear!(W)
    end
    return B
end

let src = read(joinpath(pkgdir(IP), "src", "solver.jl"), String)
    instrument(code) = replace(code, r"(\w+) (\+?)= @elapsed begin" => s"\1 \2= Main.@celapsed :\1 begin")
    # solvesdp
    i0 = findfirst("function solvesdp(\n    sdp::ClusteredLowRankSDP,", src)
    @assert i0 !== nothing "iterprof: solvesdp start"
    i1 = findnext("    status, primalsol, dualsol, time_total, error_code[1]\nend\n", src, first(i0))
    @assert i1 !== nothing "iterprof: solvesdp end"
    code = src[first(i0):last(i1)]
    a = "        println(\"A(n) \$(typeof(e)) occurred.\")\n"
    @assert count(a, code) == 1 "iterprof: error anchor"
    code = replace(code, a => a * "        try; showerror(stdout, e, catch_backtrace()); println(); catch; end; flush(stdout)\n")
    n0 = count("@elapsed begin", code)
    code = instrument(code)
    @assert count("@elapsed begin", code) == 0 && count("Main.@celapsed", code) == n0 "iterprof: solvesdp sites"
    a = "        GC.gc(false) #not sure if needed\n"
    @assert count(a, code) == 1 "iterprof: gc anchor"
    code = replace(code, a => "        Main.@celapsed :gcfalse GC.gc(false)\n")
    a = "        allocs[6]+= @allocated begin\n"
    @assert count(a, code) == 1 "iterprof: objectives anchor"
    code = replace(code, a => "        Main.@celapsed :objectives allocs[6]+= @allocated begin\n")
    a = "    # get the blocksizes to initiate the variables\n"
    @assert count(a, code) == 1 "iterprof: setup start anchor"
    code = replace(code, a => a * "    _tsetup0 = time()\n")
    a = "    leftvecs_pairings, rightvecs_pairings, pointers_left, pointers_right,high_ranks = precompute_matrices_bilinear_pairings(sdp,subblocksizes,prec=prec)\n"
    @assert count(a, code) == 1 "iterprof: precompute anchor"
    code = replace(code, a => "    _tpre = @elapsed begin\n" * a * "    end\n    println(\"TSETUP precompute_matrices_bilinear_pairings \$(round(_tpre, digits=1))s\"); flush(stdout)\n")
    a = "    #step 2\n"
    @assert count(a, code) == 1 "iterprof: setup end anchor"
    code = replace(code, a => "    println(\"TSETUP solvesdp setup \$(round(time() - _tsetup0, digits=1))s\"); flush(stdout)\n" * a)
    # one TPHASE line after the verbose iteration line (the objectives of the next iteration are timed into the
    # following line)
    a = "        # Compute the new objectives, for the new iteration\n"
    @assert count(a, code) == 1 "iterprof: print anchor"
    code = replace(code, a => "        Main._cprof_print(iter)\n" * a)
    a = "    try #catch InterruptExceptions & SolverFailures (so that we still return something)\n"
    @assert count(a, code) == 1 "iterprof: try anchor"
    code = replace(code, a => "    _bi0, _g0, _bf0 = beta_infeasible, gamma, beta_feasible\n" * a)
    a = "        mu = dot(X, Y) / size(X, 1)\n"
    @assert count(a, code) == 1 "iterprof: mu anchor"
    code = replace(code, a => a * "        beta_infeasible, gamma, beta_feasible = Main.k5_adapt(X, Y, mu, _bi0, _g0, _bf0, iter)\n")
    Base.eval(IP, Meta.parse(code))
    # compute_T_decomposition! and compute_search_direction!
    for (start, stop) in (("function compute_T_decomposition!(", "\nend\n"), ("function compute_search_direction!(", "\nend\n"))
        i0 = findfirst(start, src); @assert i0 !== nothing "iterprof: $start"
        i1 = findnext(stop, src, first(i0))
        code = src[first(i0):last(i1)]
        n0 = count("@elapsed begin", code)
        code = instrument(code)
        @assert n0 > 0 && count("Main.@celapsed", code) == n0 "iterprof: sites in $start"
        if startswith(start, "function compute_search_direction!")
            # Z and dY: the block products ran one block per task (the largest SOS block bounded the phase); the
            # tiled matmul_threaded! (blockchol.jl) also splits each large product
            for (a, b) in (("Arblib.approx_mul!(dY.blocks[j].blocks[l], P.blocks[j].blocks[l], Y.blocks[j].blocks[l])",
                            "matmul_threaded!(dY.blocks[j].blocks[l], P.blocks[j].blocks[l], Y.blocks[j].blocks[l])"),
                           ("Arblib.approx_mul!(dY.blocks[j].blocks[l], dX.blocks[j].blocks[l], Y.blocks[j].blocks[l])",
                            "matmul_threaded!(dY.blocks[j].blocks[l], dX.blocks[j].blocks[l], Y.blocks[j].blocks[l])"))
                @assert count(a, code) == 1 "iterprof: search-direction product anchor"
                code = replace(code, a => b)
            end
            # the triangular solves X^-1 (.) of Z and dY (one block per task; FLINT solves all columns serially):
            # split the right-hand-side columns of large blocks over tasks
            a = "Arblib.solve_cho_precomp!(dY.blocks[j].blocks[l], X_inv.blocks[j].blocks[l], dY.blocks[j].blocks[l])"
            @assert count(a, code) == 2 "iterprof: solve_cho_precomp anchors"
            code = replace(code, a => "Main.k5_solve_cho!(dY.blocks[j].blocks[l], X_inv.blocks[j].blocks[l])")
        end
        Base.eval(IP, Meta.parse(code))
    end
end
