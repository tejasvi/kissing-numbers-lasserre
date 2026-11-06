# Threaded versions of two ClusteredLowRankSolver 1.0.3 routines that run 3 times per iteration (residual P / dX
# for the predictor and corrector, and <A_*, Z> for d and the two right-hand sides) and were mostly serial:
#   compute_weighted_A!  sum_i x_i A_i: the stock loop visits the blocks one after another (high-rank blocks add
#                        x_p A_p sample by sample); here the blocks are tasks (largest first), each with its own
#                        scratch windows, and the low-rank products go through the tiled matmul_threaded!.
#   trace_A              <A_p, Z>: the high-rank part was a serial loop over samples and blocks; here it is
#                        threaded over samples (each sample sums its blocks, so no two tasks write one entry), and
#                        the low-rank products use the classical Arb product (fastmul.jl).
# Same arithmetic per entry as the stock code except the order in which block contributions are summed.
using ClusteredLowRankSolver
include(joinpath(@__DIR__, "fastmul.jl"))

@eval ClusteredLowRankSolver function compute_weighted_A!(initial_matrix, sdp, a, vecs_left, high_ranks)
    prec = precision(a)
    joff = zeros(Int, length(sdp.A))
    for j = 2:length(sdp.A); joff[j] = joff[j-1] + size(sdp.c[j-1], 1); end
    tasks = [(j, l) for j in eachindex(sdp.A) for l in eachindex(sdp.A[j])]
    cost(t) = size(initial_matrix.blocks[t[1]].blocks[t[2]], 1)^2 * max(1, sum(length(sdp.A[t[1]][t[2]][r, s]) for r = 1:size(sdp.A[t[1]][t[2]], 1) for s = 1:r))
    sort!(tasks, by=t -> -cost(t))
    Threads.@threads :dynamic for (j, l) in tasks
        M = initial_matrix.blocks[j].blocks[l]
        Arblib.zero!(M)
        j_idx = joff[j]
        if high_ranks[j][l]
            for p in keys(sdp.A[j][l][1, 1])
                Arblib.addmul!(M, sdp.A[j][l][1, 1][p], a[j_idx+p, 1])
            end
        else
            delta = div(size(M, 1), size(sdp.A[j][l], 1))
            cur = AL.Arb(1, prec=prec)
            vec = ArbRefMatrix(delta, 1, prec=prec)
            Q = ArbRefMatrix(0, 0, prec=prec)
            w = ArbRefMatrix(0, 0, prec=prec)
            for r = 1:size(sdp.A[j][l], 1)
                for s = 1:r
                    Arblib.window_init!(Q, M, (r-1)*delta, (s-1)*delta, r*delta, s*delta)
                    vecs_right = ArbRefMatrix(size(vecs_left[j][l][r, s], 2), size(vecs_left[j][l][r, s], 1), prec=prec)
                    idx = 1
                    for p in keys(sdp.A[j][l][r, s])
                        for rnk in eachindex(sdp.A[j][l][r, s][p].lambda)
                            Arblib.mul!(cur, a[j_idx+p, 1], sdp.A[j][l][r, s][p].lambda[rnk])
                            Arblib.mul!(vec, sdp.A[j][l][r, s][p].vs[rnk], cur)
                            Arblib.window_init!(w, vecs_right, idx-1, 0, idx, delta)
                            Arblib.transpose!(w, vec)
                            Arblib.window_clear!(w)
                            idx += 1
                        end
                    end
                    Arblib.get_mid!(vecs_right, vecs_right)
                    matmul_threaded!(Q, vecs_left[j][l][r, s], vecs_right)
                    Arblib.transpose!(Q, Q)
                    Arblib.get_mid!(Q, Q)
                    Arblib.window_clear!(Q)
                end
            end
        end
        if size(sdp.A[j][l], 1) > 1
            symmetric!(M, :L)
        end
    end
    return nothing
end

@eval ClusteredLowRankSolver function trace_A(sdp, Z::BlockDiagonal, vecs_left, vecs_right, high_ranks)
    prec = precision(Z)
    result = ArbRefMatrix(sum(size.(sdp.c, 1)), 1, prec=prec)
    Arblib.zero!(result)
    res = AL.Arb(prec=prec)
    j_idx = 0
    for j in eachindex(sdp.A)
        hls = [l for l in eachindex(sdp.A[j]) if high_ranks[j][l]]
        if !isempty(hls)
            Pset = Set{Int}()
            for l in hls; union!(Pset, keys(sdp.A[j][l][1, 1])); end
            Ps = collect(Pset)
            Threads.@threads :dynamic for p in Ps
                acc = AL.Arb(0, prec=prec)
                for l in hls
                    Ap = get(sdp.A[j][l][1, 1], p, nothing)
                    Ap === nothing && continue
                    Arblib.add!(acc, acc, dot(Ap, Z.blocks[j].blocks[l]))
                end
                Arblib.add!(result[j_idx+p], result[j_idx+p], acc)
            end
        end
        for l in eachindex(sdp.A[j])
            high_ranks[j][l] && continue
            delta = div(size(Z.blocks[j].blocks[l], 1), size(sdp.A[j][l], 1))
            ones = ArbRefMatrix(1, delta, prec=prec)
            Arblib.ones!(ones)
            for r = 1:size(sdp.A[j][l], 1)
                for s = 1:r
                    ncols = size(vecs_right[j][l][r, s], 2)
                    nchunk = max(1, min(4 * Threads.nthreads(), cld(ncols, 64)))
                    indices = [round(Int, i * ncols / nchunk) for i = 0:nchunk]
                    result_parts = [ArbRefMatrix(1, indices[i+1] - indices[i], prec=prec) for i = 1:nchunk]
                    Threads.@threads :dynamic for i = 1:nchunk
                        indices[i+1] > indices[i] || continue
                        w1 = ArbRefMatrix(0, 0, prec=prec)
                        w2 = ArbRefMatrix(0, 0, prec=prec)
                        Arblib.window_init!(w1, Z.blocks[j].blocks[l], (r-1)*delta, (s-1)*delta, r*delta, s*delta)
                        Arblib.window_init!(w2, vecs_right[j][l][r, s], 0, indices[i], size(vecs_right[j][l][r, s], 1), indices[i+1])
                        ZV = ArbRefMatrix(delta, indices[i+1] - indices[i], prec=prec)
                        Main.k5mul!(ZV, w1, w2, prec=prec)
                        Arblib.window_clear!(w1)
                        Arblib.window_clear!(w2)
                        Arblib.window_init!(w1, vecs_left[j][l][r, s], 0, indices[i], size(vecs_left[j][l][r, s], 1), indices[i+1])
                        Arblib.mul_entrywise!(ZV, ZV, w1)
                        Arblib.window_clear!(w1)
                        Main.k5mul!(result_parts[i], ones, ZV, prec=prec)
                    end
                    result_part = hcat(result_parts...)
                    idx = 1
                    for p in keys(sdp.A[j][l][r, s])
                        Arblib.zero!(res)
                        for rnk = 1:length(sdp.A[j][l][r, s][p].vs)
                            Arblib.addmul!(res, sdp.A[j][l][r, s][p].lambda[rnk], result_part[1, idx])
                            idx += 1
                        end
                        if r != s
                            Arblib.mul!(res, res, 2)
                        end
                        Arblib.add!(result[j_idx+p, 1], res, result[j_idx+p, 1])
                    end
                end
            end
        end
        j_idx += size(sdp.c[j], 1)
    end
    return result
end
