# ClusteredLowRankSolver 1.0.3's compute_step_length runs a KrylovKit Lanczos eigsolve (Float64 BLAS) for every
# block inside Threads.@threads; with ~150 blocks and 192 Julia threads, OpenBLAS (compiled for 64 threads) runs
# out of buffer regions and aborts ("too many memory regions", segfault, seen at (14,18) on 2026-10-20). The
# eigsolve calls are limited to 32 concurrent ones here, and BLAS itself is single-threaded per call.
# If eigsolve throws (LAPACKException(22) was seen on a warm start), the block falls back to the stock Arb QR
# eigenvalue path instead of aborting the solve.
# K5_STEPLOG=1: print the blocks with the most negative eig(L^-1 dM L^-T) (the ones limiting the step length)
# for every call (primal X first, then dual Y), with block names if Main.K5_BLOCKNAMES is set.
# 2026-10-21: the semaphore did not stop the aborts (OpenBLAS keeps per-thread buffers, so the number of distinct
# threads that ever called BLAS matters); the eigenvalue now comes from the pure-Julia k5_eigmin (eigmin_jl.jl),
# so no BLAS call is left in the threaded solver paths.
using ClusteredLowRankSolver, LinearAlgebra
include(joinpath(@__DIR__, "eigmin_jl.jl"))
const FB = ClusteredLowRankSolver
function k5_eig_step(M)
    F = Float64.(M)
    all(isfinite, F) || return ([0.0], nothing, (converged = 0,))   # stock Arb QR fallback
    return ([k5_eigmin!(Matrix{Float64}((F + F') / 2))], nothing, (converged = 1,))
end
const BLAS_SEM = Base.Semaphore(parse(Int, get(ENV, "K5_BLAS_CONC", "32")))
const K5_EIGFAIL = Threads.Atomic{Int}(0)
const K5_STEPCALL = Ref(0)
BLAS.set_num_threads(1)
# FLINT (under Arb's block matrix product) calls cblas_dgemm from a separate 32-bit-int OpenBLAS (OpenBLAS32_jll,
# built for 64 threads): with 192 solver threads multiplying at once it aborted with "too many memory regions"
# (the crashes of 2026-10-20/21). That library's own threading is switched off here, so a dgemm call holds one
# buffer; the library itself is replaced by a 512-thread build of the same version (see STATUS.md).
let libs = filter(l -> occursin("libopenblas", l) && !occursin("64_", l), Libc.Libdl.dllist())
    for l in libs
        try
            ccall(Libc.Libdl.dlsym(Libc.Libdl.dlopen(l), :openblas_set_num_threads), Cvoid, (Cint,), 1)
        catch
        end
    end
end

function k5_steplog(M, min_eig_arb, gamma)
    K5_STEPCALL[] += 1
    side = isodd(K5_STEPCALL[]) ? "X" : "Y"
    lst = [(BigFloat(min_eig_arb[j][l]), j, l) for j in eachindex(M.blocks) for l in eachindex(M.blocks[j].blocks)]
    sort!(lst, by=first)
    io = IOBuffer()
    print(io, "STEPLOG $side")
    for (e, j, l) in lst[1:min(6, end)]
        nm = isdefined(Main, :K5_BLOCKNAMES) ? Main.K5_BLOCKNAMES[j][l] : (j, l)
        B = M.blocks[j].blocks[l]
        Bf = Float64.(B); ev = [k5_eigmin(Bf), -k5_eigmin(-Bf)]
        print(io, " | $nm s=$(size(B, 1)) eig=", round(Float64(e), sigdigits=4),
              " step=", round(Float64(-BigFloat(gamma) / min(e, BigFloat(-1e-300))), sigdigits=3),
              " M:[", round(ev[1], sigdigits=3), ",", round(ev[end], sigdigits=3), "]")
    end
    K5_EIGFAIL[] > 0 && print(io, " | eigsolve failures so far: $(K5_EIGFAIL[])")
    println(String(take!(io))); flush(stdout)
end

let src = read(joinpath(pkgdir(FB), "src", "solver.jl"), String)
    i0 = findfirst("function compute_step_length(", src)
    i1 = findnext("\nend\n", src, first(i0))
    code = src[first(i0):last(i1)]
    a = "values, vecs, info = eigsolve("
    @assert count(a, code) == 1 "fixblas anchor"
    k = findfirst(a, code)
    e = findnext(")\n", code, last(k))
    call = code[first(k)+length("values, vecs, info = "):first(e)]
    code = code[1:first(k)-1] * "values, vecs, info = Main.k5_eig_step(tempX[2].blocks[j].blocks[l])" * code[last(e):end]
    a = "    min_eig = minimum(eigs)\n"
    @assert count(a, code) == 1 "fixblas steplog anchor"
    code = replace(code, a => a * "    haskey(ENV, \"K5_STEPLOG\") && Main.k5_steplog(M, min_eig_arb, gamma)\n")
    Base.eval(FB, Meta.parse(code))
end
