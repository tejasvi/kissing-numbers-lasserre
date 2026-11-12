# Code used to compute and round the certificates for τ5 ≤ 43, τ6 ≤ 76 and τ7 ≤ 132

These Julia files run on top of Julia 1.10 and the package LasserreSphericalCodes (with its dependency
ClusteredLowRankSolver 1.0.3). LasserreSphericalCodes comes from the data set of de Laat, Leijenhorst and de Muinck
Keizer (https://doi.org/10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594.v1). The Julia files here are included for
reference: the verification program of LasserreSphericalCodes suffices to check the certificates in `proof_5_43.tgz`,
`proof_6_76.tgz` and `proof_7_132.tgz`; none of the Julia files here is needed for the check. Paths in the shell
scripts refer to the directory layout of our runs and have to be adapted.

## Computing the zonal polynomial files
- `zonal_one.jl`, `zonal_all.sh`: compute the zonal polynomial files with the package's own routines, one process per
  partition, with many processes in parallel. The auxiliary files written by these routines do not depend on the
  dimension: the zonal files for n = 6 and n = 7 were computed from the auxiliary files made for n = 5 (the run for
  n = 6 used `KEEP_BASE=1`, so that `zonal_one.jl` kept the auxiliary files).
- `zonal_skip.patch`: our one change to the package itself, in `src/zonal_matrices_in_S.jl`. When all zonal files for
  a dimension are present, `compute_PS` returns at once instead of first writing the auxiliary files (see the
  README.md of the supplement). We used the patch for computing and rounding the solutions, as well as for one of our
  two runs of each check.

## Solving the semidefinite program
- `solve_ckpt.jl n d1 delta prec eps K [target]`: runs the interior-point solver of ClusteredLowRankSolver on the
  second level of the Lasserre hierarchy (domain [-1, cos θ], cos θ = 1/2 by default), in chunks of K iterations,
  writing the current iterate to `ckpt.jls` after each chunk and resuming from that file. Our runs used
  `env K5_PRECS=128,256 julia solve_ckpt.jl n 14 20 256 1e-12 3 -` with n = 5, 6 and 7. For n = 6 the Cholesky
  factorization of the Schur complement failed at the starting point in 128-bit arithmetic, so the first three
  iterations ran at 256 bits and the others ran at 128 bits. From each run, two iterates were then advanced by one
  iteration each in 256-bit arithmetic: iterates 96 and 117 of the run for n = 5 (with `ref256.sh`); iterates 117 and
  126 of the run for n = 6 (with `ref256_n6_v1.sh`); and iterates 99 and 132 of the run for n = 7 (with
  `ref256_n7_v1.sh`). The advanced iterates are the ones combined below.
- Further files replace routines of ClusteredLowRankSolver 1.0.3 and LasserreSphericalCodes by faster (mostly
  multithreaded) versions; each file starts with a comment that says what the file replaces. The files are
  `fastconstraint.jl`, `fastconstraint2.jl` (constraint construction); `fastsample.jl` (sampling); `fastconv.jl`
  (conversion of the problem); `fastprecomp.jl` (precomputation); `fastschur.jl`, `fastlowrank.jl`,
  `lowrank_tiled.jl`, `highrank_all.jl`, `fastmul.jl` (Schur complement); `fastresid.jl` (residuals); `blockchol.jl`
  (Cholesky); `fixblas.jl` (step lengths); `eigmin_jl.jl`, `iterprof.jl` (profiling); `k5ser.jl` (serialization);
  `cardinality.jl` (optional extra identities, not used in our runs); `k5_ozaki.jl`, `fastS_oz.jl` (an optional Schur
  complement routine, not used in our runs).

## The certificate points and the rounding
- `make_target_point.jl A.jls B.jls N W.jls`: computes the convex combination W = (1 - γ) Y_A + γ Y_B of two dual
  iterates with bounds b_A > N > b_B; W has bound exactly N (paper, Section 4). The target N is 4399//100 for n = 5,
  7699//100 for n = 6 and 13299//100 for n = 7.
- `roundsol.jl 5 -1 1//2 14 20 4399//100 W.jls`: rounds W to an exact solution with ClusteredLowRankSolver's
  `exact_solution`, writes the certificate files in the file format of LasserreSphericalCodes and runs that package's
  `verify`. `fastround.jl` holds our changes to the rounding: a Cholesky test instead of singular value
  decompositions, row operations in FLINT, a column selection that keeps every entry of the smaller blocks, a random
  prime for the modular consistency test, and the reuse of the residual polynomials when more columns are needed.
- `roundsol_inc3.jl n -1 1//2 14 20 N W.jls` (n = 6 and n = 7): works like `roundsol.jl`, but with
  `fastround_inc3.jl`, which chooses the corrected entries in batches until the linear system has a solution modulo a
  random prime (paper, Section 4). Our runs used
  `K5_INCREMENTAL=1 K5_INC_BATCH=100000 K5_INC_EXTRA=600 K5_INC_TOP=45644`.
- `cert_run2.sh` (n = 5), `cert_run6_v1.sh` (n = 6) and `cert_run7_v3.sh` (n = 7): run the combination and the
  rounding above.
