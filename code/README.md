# Code used to compute and round the certificates for τ5 ≤ 43 and τ7 ≤ 132

These Julia files run on top of the package LasserreSphericalCodes from the data set of de Laat, Leijenhorst and
de Muinck Keizer (https://doi.org/10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594.v1), its dependency
ClusteredLowRankSolver 1.0.3, and Julia 1.10. They are included for reference: the certificates in
`proof_5_43.tgz` and `proof_7_132.tgz` are checked by the verification program of that package alone, and none of
these files is needed to check them. Paths in the shell scripts refer to the directory layout of our runs and have to
be adapted.

## Computing the zonal polynomial files
- `zonal_one.jl`, `zonal_all.sh`: compute the zonal polynomial files with the package's own routines, one partition
  at a time and in parallel. The auxiliary files do not depend on the dimension: the zonal files for n = 7 were
  computed from the auxiliary files made for n = 5.
- `zonal_skip.patch`: our one change to the package itself, in `src/zonal_matrices_in_S.jl`. When all zonal files for
  the dimension are present, `compute_PS` returns at once instead of first writing the auxiliary files (see
  README.md). We used it for computing and rounding the solutions and for one of the two runs of each check.

## Solving the semidefinite program
- `solve_ckpt.jl n d1 delta prec eps K [target]`: runs the interior-point solver of ClusteredLowRankSolver on the
  second level of the hierarchy (domain [-1, cos θ], cos θ = 1/2 by default), in chunks of K iterations, writing
  the current iterate to `ckpt.jls` after each chunk and resuming from it. Our runs used
  `env K5_PRECS=128,256 julia solve_ckpt.jl n 14 20 256 1e-12 3 -` with n = 5 and n = 7. Iterates 96 and 117 of the
  run for n = 5 (with `ref256.sh`) and iterates 99 and 132 of the run for n = 7 (with `ref256_n7_v1.sh`) were then
  advanced by one iteration each in 256-bit arithmetic; they are the iterates combined below.
- The other files replace routines of ClusteredLowRankSolver 1.0.3 and LasserreSphericalCodes by faster
  (mostly multithreaded) versions; each file starts with a comment that says what it replaces: `fastconstraint.jl`,
  `fastconstraint2.jl` (constraint construction), `fastsample.jl` (sampling), `fastconv.jl` (conversion of the
  problem), `fastprecomp.jl` (precomputation), `fastschur.jl`, `fastlowrank.jl`, `lowrank_tiled.jl`,
  `highrank_all.jl`, `fastmul.jl` (Schur complement), `fastresid.jl` (residuals), `blockchol.jl` (Cholesky),
  `fixblas.jl` (step lengths), `eigmin_jl.jl`, `iterprof.jl` (profiling), `k5ser.jl` (serialization), `cardinality.jl`
  (optional extra identities, not used here), `k5_ozaki.jl`, `fastS_oz.jl` (an optional Schur complement, not used
  here).

## The certificate points and the rounding
- `make_target_point.jl A.jls B.jls N W.jls`: the convex combination W = (1 - γ) Y_A + γ Y_B of two dual iterates
  with bounds b_A > N > b_B, which has bound exactly N (paper, Section 4); N = 4399//100 for n = 5 and 13299//100 for
  n = 7.
- `roundsol.jl 5 -1 1//2 14 20 4399//100 W.jls`: rounds W to an exact solution with ClusteredLowRankSolver's
  `exact_solution`, writes the certificate files in the package's file format and runs its `verify`.
  `fastround.jl` holds our changes to the rounding: a Cholesky test instead of singular value decompositions, row
  operations in FLINT, a column selection that keeps every entry of the smaller blocks, a random prime for the
  modular consistency test, and the reuse of the residual polynomials when more columns are needed.
- `roundsol_inc3.jl 7 -1 1//2 14 20 13299//100 W.jls`, run with `K5_INCREMENTAL=1 K5_INC_BATCH=100000
  K5_INC_EXTRA=600 K5_INC_TOP=45644`: the same with `fastround_inc3.jl`, which chooses the corrected entries in
  batches until the linear system has a solution modulo a random prime (paper, Section 4).
- `cert_run2.sh` (n = 5) and `cert_run7_v3.sh` (n = 7): run the two steps above.
