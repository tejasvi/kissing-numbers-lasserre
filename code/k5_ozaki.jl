# load the Ozaki Schur complement into ClusteredLowRankSolver (replaces compute_S_integrated!)
using ClusteredLowRankSolver
Base.include(ClusteredLowRankSolver, joinpath(@__DIR__, "fastS_oz.jl"))
println("Ozaki Schur complement loaded: K=$(ClusteredLowRankSolver.OZ_K[]) L=$(ClusteredLowRankSolver.OZ_LEVELS[])"); flush(stdout)
