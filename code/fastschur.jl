# Faster Schur-complement contribution of HIGH-RANK blocks in ClusteredLowRankSolver.
# The original loops over samples p sequentially: T = X^{-1} A_p Y (small matmuls), then
# S[p,q] += <A_q, T> for q >= p (element-wise dot products). For the level-2 zonal blocks
# (dense s x s matrices at m ~ 4000 samples) this serial loop dominates the iteration time.
# Here the same quantity is computed with three large threaded matrix products:
#   T1 = B [A_1 ... A_m] (B = X^{-1}),  T2 = (stacked T1 blocks) * Y,  G = M^T W,
# with M = [vec A_p], W = [vec(B A_p Y)], and S[p,q] += G[p,q] for q >= p.
using ClusteredLowRankSolver
const CLRS2 = ClusteredLowRankSolver
include(joinpath(@__DIR__, "fastlowrank.jl"))
include(joinpath(@__DIR__, "lowrank_tiled.jl"))
include(joinpath(@__DIR__, "highrank_all.jl"))
const ArbM = CLRS2.Arblib.ArbRefMatrix
const ArbL = CLRS2.Arblib

function _w(A, r1, c1, r2, c2)
    W = ArbM(0, 0, prec=precision(A))
    ArbL.window_init!(W, A, r1, c1, r2, c2)
    W
end

function highrank_contrib!(S, sdp, X_inv, Y, tempX, j, l, matmul_prec, prec)
    Ad = sdp.A[j][l][1, 1]
    Ps = sort(collect(keys(Ad)))
    m = length(Ps)
    m == 0 && return
    s = size(Y.blocks[j].blocks[l], 1)
    Binv = tempX[1].blocks[j].blocks[l]
    ArbL.inv_cho_precomp!(Binv, X_inv.blocks[j].blocks[l])
    ArbL.get_mid!(Binv, Binv)
    Yb = Y.blocks[j].blocks[l]
    Acat = ArbM(s, s * m, prec=prec)
    Mm = ArbM(s * s, m, prec=prec)
    Threads.@threads for a = 1:m
        Ap = Ad[Ps[a]]
        W1 = _w(Acat, 0, (a - 1) * s, s, a * s)
        ArbL.set!(W1, Ap)
        for c = 1:s, r = 1:s
            Mm[(c - 1) * s + r, a] = Ap[r, c]
        end
    end
    T1 = ArbM(s, s * m, prec=prec)
    CLRS2.matmul_threaded!(T1, Binv, Acat, prec=matmul_prec)
    ArbL.get_mid!(T1, T1)
    T1v = ArbM(s * m, s, prec=prec)
    Threads.@threads for a = 1:m
        ArbL.set!(_w(T1v, (a - 1) * s, 0, a * s, s), _w(T1, 0, (a - 1) * s, s, a * s))
    end
    T2v = ArbM(s * m, s, prec=prec)
    CLRS2.matmul_threaded!(T2v, T1v, Yb, prec=matmul_prec)
    ArbL.get_mid!(T2v, T2v)
    Wm = ArbM(s * s, m, prec=prec)
    Threads.@threads for a = 1:m
        for c = 1:s, r = 1:s
            Wm[(c - 1) * s + r, a] = T2v[(a - 1) * s + r, c]
        end
    end
    MmT = ArbM(m, s * s, prec=prec)
    ArbL.transpose!(MmT, Mm)
    G = ArbM(m, m, prec=prec)
    CLRS2.matmul_threaded!(G, MmT, Wm, prec=matmul_prec)
    ArbL.get_mid!(G, G)
    Sj = S[j]
    # S[p,q] += <A_q, X^{-1} A_p Y> = G[b, a] for q = Ps[b] >= p = Ps[a]; rows of S are independent
    Threads.@threads for a = 1:m
        p = Ps[a]
        for b = a:m
            ArbL.add!(Sj[p, Ps[b]], Sj[p, Ps[b]], G[b, a])
        end
    end
    return
end

let src = read(joinpath(pkgdir(CLRS2), "src", "solver.jl"), String)
    i = findfirst("function compute_S_integrated!(", src)
    k = findnext("\"\"\"Compute the decomposition of [S B; B^T 0]\"\"\"", src, first(i))
    code = src[first(i):first(k)-1]
    h1 = findfirst("            if high_ranks[j][l]\n", code)
    h2 = findfirst("            else # low rank case", code)
    @assert h1 !== nothing && h2 !== nothing
    newcode = code[1:first(h1)-1] *
        "            if high_ranks[j][l]\n                Main.highrank_contrib!(S, sdp, X_inv, Y, tempX, j, l, matmul_prec, prec)\n" *
        code[first(h2):end]
    # get_mid! of the whole preallocated pairing matrices (sized for the largest block) after every block
    # cost ~2 x 1e8 entry operations per block; apply it to the filled window instead (approx_mul! output
    # has zero radius anyway)
    for nm in ("bilinear_pairings_Y", "bilinear_pairings_Xinv")
        old1 = "                        Arblib.window_clear!(temp_window)\n                        Arblib.window_clear!(temp_window2)\n                        Arblib.get_mid!($nm[s,r], $nm[s,r])\n"
        old2 = "                        Arblib.window_clear!(temp_window)\n                        Arblib.window_clear!(temp_window2)\n                        Arblib.get_mid!($nm[s,r],$nm[s,r])\n"
        new = "                        Arblib.get_mid!(temp_window2, temp_window2)\n                        Arblib.window_clear!(temp_window)\n                        Arblib.window_clear!(temp_window2)\n"
        n0 = count(old1, newcode) + count(old2, newcode)
        @assert n0 == 1 "get_mid anchor for $nm found $n0 times"
        newcode = replace(newcode, old1 => new, old2 => new)
    end
    if get(ENV, "K5_FASTLOWRANK", "1") == "1"
        # replace the O(m^2) Dict-based loop of the low-rank branch by Main.lowrank_accum! (single subblock only)
        c1 = findfirst("                # We compute the contribution of this l to S[j]", newcode)
        c2 = findfirst("            # We call the gc when the memory usage gets too large.", newcode)
        @assert c1 !== nothing && c2 !== nothing "fastlowrank anchors not found"
        body = newcode[first(c1):first(c2)-1]
        # the original loop body ends with the closing 'end' of the low-rank else-branch; keep that 'end'
        k = findlast("            end\n", body)
        @assert k !== nothing
        loop = body[1:first(k)-1]
        newloop = "                if size(sdp.A[j][l]) == (1,1)\n                    Main.lowrank_accum!(S, sdp, bilinear_pairings_Xinv, bilinear_pairings_Y, pointers_left, pointers_right, j, l, prec)\n                else\n" * loop * "                end\n"
        newcode = newcode[1:first(c1)-1] * newloop * body[first(k):end] * newcode[first(c2):end]
    end
    if get(ENV, "K5_LRTILED", "1") == "1"
        # symmetric single-subblock low-rank blocks: tiled Gram products folded straight into S (lowrank_tiled.jl)
        e0 = "            else # low rank case"
        @assert count(e0, newcode) == 1 "low-rank branch anchor"
        newcode = replace(newcode, e0 => "            elseif size(sdp.A[j][l]) == (1,1) && Main.lr_symmetric(sdp.A[j][l][1,1])\n                Main.lowrank_tiled!(S, A_Y, sdp, X_inv, Y, j, l, prec)\n" * e0)
    end
    if get(ENV, "K5_HRALL", "1") == "1"
        # all high-rank blocks of the cluster in one stacked Gram product (highrank_all.jl), at the first one
        h0 = "Main.highrank_contrib!(S, sdp, X_inv, Y, tempX, j, l, matmul_prec, prec)"
        @assert count(h0, newcode) == 1 "high-rank call anchor"
        newcode = replace(newcode, h0 => "(l == findfirst(high_ranks[j]) && Main.highrank_all!(S, sdp, X_inv, Y, j, high_ranks, prec))")
    end
    # the stock loop runs a full GC + malloc_trim whenever free memory dropped 5 GB since the start of the
    # Schur complement; with the tiled paths that fires after almost every block (millions of Arb finalizers)
    g0 = "if Sys.free_memory()/1024^3 < base - 5"
    if count(g0, newcode) == 1
        newcode = replace(newcode, g0 => "if Sys.free_memory()/1024^3 < base - parse(Float64, get(ENV, \"K5_GC_GB\", \"100\"))")
    end
    # the full GC + malloc_trim at the start of every Schur complement: optional (K5_SCHUR_GC=0 skips it) and
    # timed (K5_LRLOG); the GC share of the Schur phase was 40% at (16,16) on 192 threads
    g1 = "        GC.gc()\n        malloc_trim(0)\n    end\n    base = Sys.free_memory()/1024^3 #in GB\n"
    @assert count(g1, newcode) == 1 "start-of-Schur GC anchor"
    newcode = replace(newcode, g1 => "        if get(ENV, \"K5_SCHUR_GC\", \"1\") == \"1\"\n            _tg = time(); GC.gc(); _tg1 = time(); malloc_trim(0)\n            haskey(ENV, \"K5_LRLOG\") && (println(\"SCHURGC start gc=\$(round(_tg1 - _tg, digits=2))s trim=\$(round(time() - _tg1, digits=2))s\"); flush(stdout))\n        end\n    end\n    base = Sys.free_memory()/1024^3 #in GB\n")
    g2 = "                GC.gc() #GC.gc(false) doesn't work very well when called frequently."
    k2 = findfirst(g2, newcode)
    if k2 !== nothing
        e2 = findnext("malloc_trim(0)\n", newcode, last(k2))
        newcode = newcode[1:first(k2)-1] * "                _tg = time(); GC.gc(); _tg1 = time(); malloc_trim(0)\n                haskey(ENV, \"K5_LRLOG\") && (println(\"SCHURGC block gc=\$(round(_tg1 - _tg, digits=2))s trim=\$(round(time() - _tg1, digits=2))s\"); flush(stdout))\n" * newcode[last(e2)+1:end]
    end
    Base.eval(CLRS2, Meta.parse(newcode))
end
