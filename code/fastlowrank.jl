# Low-rank part of the Schur complement in ClusteredLowRankSolver.compute_S_integrated!, without Dict
# lookups in the O(m^2) loop. The original code runs, for every pair of samples (p, q),
#   S[p,q] += sum_{rnk1,rnk2} lambda_p lambda_q BX[L(p), R(q)] BY[L(q), R(p)]
# looking up L and R in Dicts keyed by tuples and iterating over Dict keys; here the column indices and
# eigenvalues are gathered into arrays first and the rows of S are processed with dynamic threading.
# Only blocks with a single subblock (size(sdp.A[j][l]) == (1,1)) take this path; others use the original loop.
using ClusteredLowRankSolver
const CLRS3 = ClusteredLowRankSolver
const AL3 = CLRS3.Arblib

function lowrank_accum!(S, sdp, bpX, bpY, ptrL, ptrR, j, l, prec)
    D = sdp.A[j][l][1, 1]
    Ps = sort(collect(keys(D)))
    m = length(Ps)
    m == 0 && return
    lam = [D[p].lambda for p in Ps]
    Lc = [[ptrL[j][l][1][(1, p, r)] for r in eachindex(D[p].lambda)] for p in Ps]
    Rc = [[ptrR[j][l][1][(1, p, r)] for r in eachindex(D[p].lambda)] for p in Ps]
    ones_ = [[AL3.isone(x) for x in lam[a]] for a in 1:m]
    BX = bpX[1, 1]; BY = bpY[1, 1]
    Sj = S[j]
    Threads.@threads :dynamic for a = 1:m
        tot = AL3.Arb(prec=prec)
        p = Ps[a]
        @inbounds for b = a:m
            q = Ps[b]
            s = Sj[p, q]
            for r1 in eachindex(Lc[a]), r2 in eachindex(Lc[b])
                if ones_[a][r1] && ones_[b][r2]
                    AL3.addmul!(s, BX[Lc[a][r1], Rc[b][r2]], BY[Lc[b][r2], Rc[a][r1]])
                else
                    AL3.mul!(tot, lam[a][r1], lam[b][r2])
                    AL3.mul!(tot, tot, BX[Lc[a][r1], Rc[b][r2]])
                    AL3.addmul!(s, tot, BY[Lc[b][r2], Rc[a][r1]])
                end
            end
        end
    end
    return
end
