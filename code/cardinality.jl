# fixed-N cardinality multipliers for the level-2 problem (from work/runs/card/las2card.jl); see the
# header there for the derivation. add_cardinality!(problem, N, delta) adds free variables g1, g2(u), g3(u,v,t).
using Nemo, ClusteredLowRankSolver
cheb(j, x) = j == 0 ? one(x) : (j == 1 ? x : begin a = one(x); b = x; for _ = 2:j; a, b = b, 2x*b - a; end; b end)
# basis of S3-symmetric polynomials of degree <= D: e1^a e2^b e3^c with a + 2b + 3c <= D (Chebyshev-ish in e1)
sym3basis(D) = [(a,b,c) for deg = 0:D for c = 0:div(deg,3) for b = 0:div(deg-3c,2) for a in [deg-3c-2b]]
function sym3eval(abc, u, v, t)
    a, b, c = abc
    e1 = u + v + t; e2 = u*v + v*t + u*t; e3 = u*v*t
    cheb(a, e1/3) * e2^b * e3^c
end

function add_cardinality!(problem, N, delta; g2deg=delta, g3deg=delta)
    cons = problem.constraints
    @assert length(cons) >= 4
    Nq = Rational{BigInt}(N)
    b3 = sym3basis(g3deg)
    for (k, c) in enumerate(cons[1:4])
        S = c.samples
        R = SampledMPolyRing(QQ, S)
        mk(ev) = SampledMPolyRingElem(R, [QQ(x) for x in ev])
        if k == 1
            c.freecoeff[:g1] = mk([-(Nq-1) for s in S])
        elseif k == 2
            c.freecoeff[:g1] = mk([2 for s in S])
            for j = 0:g2deg
                c.freecoeff[(:g2, j)] = mk([-(Nq-2) * cheb(j, Rational{BigInt}(s[1])) for s in S])
            end
        elseif k == 3
            for j = 0:g2deg
                c.freecoeff[(:g2, j)] = mk([sum(cheb(j, Rational{BigInt}(s[i])) for i = 1:3) for s in S])
            end
            for (bi, abc) in enumerate(b3)
                c.freecoeff[(:g3, bi)] = mk([-(Nq-3) * sym3eval(abc, Rational{BigInt}.(s)...) for s in S])
            end
        elseif k == 4
            for (bi, abc) in enumerate(b3)
                ev = []
                for s in S
                    x = Rational{BigInt}.(s)
                    push!(ev, sym3eval(abc, x[1], x[2], x[4]) + sym3eval(abc, x[1], x[3], x[5]) +
                              sym3eval(abc, x[2], x[3], x[6]) + sym3eval(abc, x[4], x[5], x[6]))
                end
                c.freecoeff[(:g3, bi)] = mk(ev)
            end
        end
    end
    problem
end

