# The kissing numbers in dimensions five and seven are at most 43 and 132

This repository holds the paper "The kissing numbers in dimensions five and seven are at most 43 and 132" by
Tejasvi S. Tomar, its source in one LaTeX file, and the Julia code used to compute and round its certificates. The
paper is available at https://doi.org/10.5281/zenodo.23992400. The certificates, the zonal polynomial files and the
Lean 4 project are attached as assets to release v2 of this repository, and the same files are in the supplement on
Zenodo, https://doi.org/10.5281/zenodo.23992399.

| path | content |
|---|---|
| `paper/main.pdf`, `paper/main.tex` | the paper and its source, including the bibliography |
| `code/` | the Julia code, with its own README.md |

| release asset | content |
|---|---|
| `proof_5_43.tgz` | certificate for τ5 ≤ 43 (n = 5, cos θ = 1/2, degrees (14, 20, 20), bound 4399/100) |
| `proof_7_132.tgz` | certificate for τ7 ≤ 132 (n = 7, cos θ = 1/2, degrees (14, 20, 20), bound 13299/100) |
| `zonal_5.tgz`, `zonal_7.tgz` | the zonal polynomial files needed for the checks |
| `lean_proofs.tar.gz` | the Lean 4 project, with its own README.md |
| `code.tgz` | the contents of `code/` |
| `README.md` | the README of the supplement: how to check the certificates and build the Lean proofs |
| `SHA256SUMS` | checksums of the files above |

The certificates are checked by the verification program in the data set of D. de Laat, N. Leijenhorst and W. de
Muinck Keizer, https://doi.org/10.4121/74ce1c25-6fca-4680-8a36-e9c18e7e9594.v1, as the supplement's README.md
explains.

To cite the paper, see `CITATION.cff`. The paper and the code are licensed under the Creative Commons Attribution
4.0 International license (`LICENSE-CC-BY-4.0.txt`), the same license as on Zenodo. Copyright 2026 Tejasvi S.
Tomar.
