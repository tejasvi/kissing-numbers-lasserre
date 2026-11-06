# Level-2 SDP solve with checkpoints (survives spot preemption): runs solvesdp in chunks of K
# iterations, serializing the current iterate (primal, dual) to ckpt.jls after each chunk, and warm
# starts from it when restarted.
using LasserreSphericalCodes, ClusteredLowRankSolver, Serialization
const L = LasserreSphericalCodes
const CLRS = ClusteredLowRankSolver
include(joinpath(@__DIR__, "k5ser.jl"))   # binary Arb matrix serialization for the SDP caches (reads the old format too)
include(joinpath(@__DIR__, "blockchol.jl"))
include(joinpath(@__DIR__, "fixblas.jl"))   # limit concurrent BLAS calls in the step-length computation
include(joinpath(@__DIR__, "iterprof.jl"))   # per-phase profile, adaptive centering, nested errors (supersedes showerr.jl)
include(joinpath(@__DIR__, "fastconstraint.jl"))
get(ENV, "K5_FC", "2") == "2" && include(joinpath(@__DIR__, "fastconstraint2.jl"))   # in-place Arb zonal evaluation
get(ENV, "K5_CONV", "fast") == "fast" && include(joinpath(@__DIR__, "fastconv.jl"))      # threaded ClusteredLowRankSDP conversion
if get(ENV, "K5_SCHUR", "arb") == "ozaki"
    include(joinpath(@__DIR__, "k5_ozaki.jl"))
else
    include(joinpath(@__DIR__, "fastschur.jl"))
end
include(joinpath(@__DIR__, "cardinality.jl"))
include(joinpath(@__DIR__, "fastsample.jl"))   # threaded 4-point SOS sampling (setup at delta >= 22)
get(ENV, "K5_FASTRESID", "1") == "1" && include(joinpath(@__DIR__, "fastresid.jl"))   # threaded sum x_i A_i and <A_*, Z>
get(ENV, "K5_PRECOMP", "fast") == "fast" && include(joinpath(@__DIR__, "fastprecomp.jl"))   # hash-based vector dedup

n = parse(Int, ARGS[1]); d1 = parse(Int, ARGS[2]); dl = parse(Int, ARGS[3]); prec = parse(Int, ARGS[4])
eps = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1e-10
K = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 4
obj_target = length(ARGS) >= 7 && ARGS[7] != "-" ? parse(Rational{BigInt}, ARGS[7]) : nothing
# certificate (feasibility) solves: the bound constraint pins one entry of block [0,0], which then always looks
# off-centre to the adaptive centering of iterprof.jl; raising beta for it stalled every such solve on 2026-10-21
# (A(5, acos 1/3) (12,12) at 22.95: converges with K5_ADAPT=0, stalls with 1). Off by default when a target is set.
obj_target !== nothing && !haskey(ENV, "K5_ADAPT") && (ENV["K5_ADAPT"] = "0")
# precision ladder (K5_PRECS="128,256"): run at the lowest precision not yet marked failed (file
# prec_failed_P); on a Cholesky failure of the Schur complement or when the step schedule gives up, mark it and
# exit with code 3 so the job loop resumes from the checkpoint at the next precision
const PRECS = [parse(Int, x) for x in split(get(ENV, "K5_PRECS", string(prec)), ",")]
prec = something(findfirst(p -> !isfile("prec_failed_$p"), PRECS), length(PRECS)) |> i -> PRECS[i]
println("working precision $prec (ladder $(PRECS))"); flush(stdout)
setprecision(BigFloat, prec)
haskey(ENV, "K5_GCLOG") && GC.enable_logging(true)   # print every GC with its pause time

t0 = time()
# the converted SDP is cached in sdp.jls (with the objective), so a restart after preemption skips the setup
sdpfile = prec == 256 && isfile("sdp.jls") ? "sdp.jls" : "sdp_p$(prec).jls"
if isfile(sdpfile)
    objv, sdp = deserialize(sdpfile)
    println("SDP loaded from $sdpfile in $(round(time() - t0, digits=1))s"); flush(stdout)
elseif prec < 256 && (isfile("sdp.jls") || isfile("sdp_p256.jls"))
    objv, sdp = deserialize(isfile("sdp.jls") ? "sdp.jls" : "sdp_p256.jls")
    sdp = CLRS.convert_to_prec(sdp, prec)
    println("SDP loaded from the 256-bit cache and converted to $prec bits in $(round(time() - t0, digits=1))s"); flush(stdout)
elseif any(q -> q < prec && isfile("sdp_p$(q).jls"), (128, 192))
    # moving up the precision ladder without a cache at this precision: reuse the lower-precision cache instead of
    # rebuilding (hours at delta >= 18). Its entries were rounded at the lower precision, a relative perturbation of
    # the data of 2^-128 that is far below what the solve or the exact rounding (from the unsampled problem) can see
    f = first(filter(isfile, ["sdp_p$(q).jls" for q in (192, 128) if q < prec]))
    objv, sdp = deserialize(f)
    sdp = CLRS.convert_to_prec(sdp, prec)
    println("SDP loaded from $f and converted up to $prec bits in $(round(time() - t0, digits=1))s"); flush(stdout)
    global k5_converted_up = true      # no extra cache for the converted copy (conversion takes seconds)
else
    cosv = parse(Rational{BigInt}, get(ENV, "K5_COS", "1//2"))   # domain [lov, cosv] of the inner products
    lov = parse(Rational{BigInt}, get(ENV, "K5_LO", "-1"))
    problem = L.las2_problem(n, [lov, cosv], d1; delta=dl, d2=dl, obj=obj_target, verbose=true)
    println("domain [$lov, $cosv]"); flush(stdout)
    if haskey(ENV, "K5_CARD_N")   # exact-size-N identities (valid only for codes of size exactly N)
        add_cardinality!(problem, parse(Rational{BigInt}, ENV["K5_CARD_N"]), dl)
        println("cardinality identities added with N = $(ENV["K5_CARD_N"])"); flush(stdout)
    end
    objv = problem.objective
    sdp = CLRS.ClusteredLowRankSDP(problem)
    sdp = CLRS.convert_to_prec(sdp, prec)
    problem = nothing; GC.gc()
    println("SETUP done in $(round(time() - t0, digits=1))s"); flush(stdout)
    try
        free_gb = parse(Float64, split(read(`df --output=avail -BG .`, String))[end][1:end-1])
        rss_gb = Sys.maxrss() / 2^30
        if free_gb > 0.2 * rss_gb + 30   # sdp.jls is a few % of the RSS (3.3 GB at (16,16))
            t1 = time(); serialize("sdp.jls.tmp", (objv, sdp)); mv("sdp.jls.tmp", sdpfile, force=true)
            println("SDP cached to sdp.jls in $(round(time() - t1, digits=1))s"); flush(stdout)
        else
            println("not caching the SDP: free $(free_gb) GB, rss $(round(rss_gb, digits=1)) GB"); flush(stdout)
        end
    catch e
        println("SDP caching failed: $e"); rm("sdp.jls.tmp", force=true)
    end
end

# keep a binary-format cache at the working precision (loads ~15x faster than Arblib's string format): written
# when the SDP came from an old-format file or was converted from the 256-bit cache
binfile = prec == 256 && isfile("sdp.jls") ? "sdp.jls" : "sdp_p$(prec).jls"   # same name rule as sdpfile above
if !(@isdefined(k5_converted_up) && k5_converted_up) && (K5_SER_OLD[] || !isfile(binfile))
    try
        free_gb = parse(Float64, split(read(`df --output=avail -BG .`, String))[end][1:end-1])
        if free_gb > 60
            t1 = time(); serialize(binfile * ".tmp", (objv, sdp)); mv(binfile * ".tmp", binfile, force=true)
            println("SDP cache written in binary format to $binfile in $(round(time() - t1, digits=1))s"); flush(stdout)
        else
            println("not writing the binary SDP cache: free $(free_gb) GB"); flush(stdout)
        end
    catch e
        println("binary SDP cache failed: $e"); rm(binfile * ".tmp", force=true)
    end
end
const K5_BLOCKNAMES = sdp.matrix_coeff_names   # block names for the CENT / STEPLOG lines
ck = "ckpt.jls"
ps = nothing; ds = nothing; iters = 0
if isfile(ck)
    ps, ds, iters = deserialize(ck)
    println("RESUME from checkpoint after $iters iterations"); flush(stdout)
end
global status = nothing
# step-collapse handling: solvesdp aborts (err=4) when a step length falls below K5_STEP_MIN (default 1e-3;
# healthy runs stay above ~0.05), so the collapse is caught at its first iteration. The chunk is then rolled
# back and retried with a smaller step fraction gamma and more centering (beta); after the third level the
# rollback also goes one further good iterate back (a short history is kept in memory, and the previous
# checkpoint on disk as ckpt_prev.jls). The failed iterate is kept in ckpt_failed.jls. Gamma/beta drift back
# to the defaults after 3 good chunks.
const SCHED = [(9//10, 3//10), (6//10, 3//10), (6//10, 6//10), (4//10, 8//10), (25//100, 9//10)]
const STEPMIN = parse(Float64, get(ENV, "K5_STEP_MIN", "1e-3"))
global level = haskey(ENV, "K5_LEVEL") ? parse(Int, ENV["K5_LEVEL"]) : 1
global good = 0
global hist = Any[(ps, ds, iters)]   # good iterates, newest last
while true
    gam, bet = SCHED[level]
    omega = BigFloat(parse(Float64, get(ENV, "K5_OMEGA", "1e10")))   # initial point scale (cold start only)
    local st, ps2, ds2, tt, err
    try
        st, ps2, ds2, tt, err = solvesdp(sdp; prec=prec, maxiterations=K, primalsol=ps, dualsol=ds,
            duality_gap_threshold=eps, primal_error_threshold=eps, dual_error_threshold=eps,
            gamma=gam, beta_infeasible=bet, step_length_threshold=STEPMIN, omega_p=omega, omega_d=omega)
    catch e
        if prec < PRECS[end] && (e isa CLRS.SolverFailure || occursin("decomposed", sprint(showerror, e)))
            touch("prec_failed_$prec")
            println("PRECISION FAILURE at $prec bits (iters=$iters): $(sprint(showerror, e)); moving up"); flush(stdout)
            exit(3)
        end
        rethrow()
    end
    if err == 1   # solvesdp caught an exception (e.g. the Schur complement was not decomposed): keep the checkpoint
        # first error at this precision: retry it once (other failures, e.g. a failed task, look the same)
        if !isfile("err1_$prec")
            touch("err1_$prec")
            println("SOLVER ERROR at $prec bits (iters=$iters); retrying once at the same precision"); flush(stdout)
            exit(5)
        end
        if prec < PRECS[end]
            touch("prec_failed_$prec")
            println("PRECISION FAILURE at $prec bits (iters=$iters, solver error); moving up"); flush(stdout)
            exit(3)
        end
        println("SOLVER ERROR at $prec bits (iters=$iters); checkpoint kept, exiting"); flush(stdout)
        exit(4)
    end
    if err == 4 && ps !== nothing
        serialize("ckpt_failed.jls", (ps2, ds2, iters))
        if level < length(SCHED)
            global level += 1; global good = 0
            if level >= 4 && length(hist) >= 2
                pop!(hist)
                global ps, ds, iters = hist[end]
            end
            println("ROLLBACK at iters=$iters (step < $STEPMIN); retry with gamma=$(Float64(SCHED[level][1])) beta=$(Float64(SCHED[level][2]))"); flush(stdout)
            continue
        end
        if prec < PRECS[end]
            touch("prec_failed_$prec")
            println("GIVING UP at $prec bits (iters=$iters); moving up a precision"); flush(stdout)
            exit(3)
        end
        println("GIVING UP at iters=$iters after all step schedules"); flush(stdout)
        global status = st
        break
    end
    global iters += K
    isfile(ck) && mv(ck, "ckpt_prev.jls", force=true)
    serialize(ck * ".tmp", (ps2, ds2, iters)); mv(ck * ".tmp", ck, force=true)
    global ps = ps2; global ds = ds2; global status = st
    push!(hist, (ps, ds, iters)); length(hist) > 3 && popfirst!(hist)
    obj = objvalue(objv, ds)
    println("CHUNK iters=$iters status=$(typeof(st)) err=$err obj=$(Float64(obj)) gamma=$(Float64(gam)) wall=$(round(time() - t0, digits=1))s"); flush(stdout)
    if err != 2 || iters >= 600   # 2 = maxiterations of this chunk reached; anything else = finished or failed
        break
    end
    global good += 1
    if level > 1 && good >= 3
        global level -= 1; global good = 0
    end
end
obj = objvalue(objv, ds)
println("RESULT n=$n d1=$d1 d2=delta=$dl prec=$prec status=$(typeof(status)) obj=$(Float64(obj)) wall=$(time() - t0)s")
open("result.txt", "a") do f
    println(f, "n=$n d1=$d1 d2=delta=$dl prec=$prec eps=$eps target=$(obj_target) status=$(typeof(status)) obj=$(obj) iters=$iters wall=$(time() - t0)s")
end
serialize("solution.jls", (ps, ds))
