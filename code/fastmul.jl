# Arb matrix products: FLINT/Arb's arb_mat_approx_mul (block algorithm: converts blocks of entries with similar
# exponents to integer matrices) scales badly across threads when the inner dimension is long or the exponents
# are spread out: on 2026-10-21, (4096 x 10739) x (10739 x 4096) at 128 bits in 256 x 256 tiles on 192 threads
# took 204 s (block) vs 9.8 s with arb_mat_approx_mul_classical (dot products, no temporaries); with inner
# dimension 200 and exponents spread over +-100 bits, 2.5 s vs 1.2 s. libarb exports the classical routine but
# Arblib does not wrap it; k5mul! picks it by K5_MULALG = auto (default: classical when the inner dimension is
# >= K5_MULK, default 2048) | classical | block. With short inner dimensions the block algorithm is as fast or
# faster on the real data: (16,16) low-rank tiles at 128 bits 185 s block vs 195 s classical; the whole Schur
# complement of a 256-bit (12,12) problem 34.6 s block vs 47.2 s classical.
using ClusteredLowRankSolver
if !isdefined(@__MODULE__, :K5_ALGSTAT)   # included from several files; define once
const FM = ClusteredLowRankSolver.Arblib
isdefined(FM, :approx_mul_classical!) ||
    @eval FM arbcall"void arb_mat_approx_mul_classical(arb_mat_t res, const arb_mat_t mat1, const arb_mat_t mat2, slong prec)"
const K5_MULALG = get(ENV, "K5_MULALG", "auto")
const K5_MULK = parse(Int, get(ENV, "K5_MULK", "2048"))
@inline function k5mul!(C, A, B; prec=precision(C))
    if K5_MULALG == "classical" || (K5_MULALG == "auto" && size(A, 2) >= K5_MULK)
        FM.approx_mul_classical!(C, A, B, prec=prec)
    else
        FM.approx_mul!(C, A, B, prec=prec)
    end
    C
end

# Per-site choice by measurement: a site (e.g. the tile products of one low-rank block) runs its whole parallel
# phase with the untested algorithm first, then with the faster one; the slower one is re-timed only if K5_RETEST=N is set (every N calls)
# (the data drift over iterations). Thread scaling, not single-thread speed, decides, hence whole-phase timing.
const K5_ALGSTAT = Dict{Any,Vector{Float64}}()   # site => [t_block, t_classical, calls]
const K5_ALGLOCK = ReentrantLock()
function k5alg(site)
    K5_MULALG == "auto" || return K5_MULALG == "classical"
    # at 128 bits the classical product won every measured site at delta >= 16 in steady state, and re-timing the
    # block product (FLINT temporaries through a contended allocator) stalled a delta = 20 Schur complement for
    # > 90 min on 2026-10-22: no testing at <= 128 bits
    isdefined(Main, :prec) && Main.prec isa Integer && Main.prec <= 128 && return true
    lock(K5_ALGLOCK) do
        v = get!(() -> [NaN, NaN, 0.0], K5_ALGSTAT, site)
        v[3] += 1
        first_classical = site isa Tuple && length(site) >= 4 && site[4] >= K5_MULK   # long inner dimension: try classical first
        if isnan(v[1]) && isnan(v[2])
            return first_classical
        end
        isnan(v[1]) && return false
        isnan(v[2]) && return true
        fast = v[2] < v[1]
        retest = parse(Int, get(ENV, "K5_RETEST", "0"))   # 0: never re-time the slower one
        return retest > 0 && mod(v[3], retest) == 0 ? !fast : fast
    end
end
function k5alg_done!(site, classical, t)
    K5_MULALG == "auto" || return
    lock(K5_ALGLOCK) do
        v = get!(() -> [NaN, NaN, 0.0], K5_ALGSTAT, site)   # get!: k5alg returns early at <= 128 bits
        v[classical ? 2 : 1] = t
    end
end
@inline function k5mul!(C, A, B, classical::Bool; prec=precision(C))
    classical ? FM.approx_mul_classical!(C, A, B, prec=prec) : FM.approx_mul!(C, A, B, prec=prec)
    C
end
end
