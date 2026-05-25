using DrWatson
@quickactivate "project"
include(srcdir("sir_model.jl"))

# Import required packages
using Random
using StatsPlots
using BenchmarkTools
using DataFrames

tmax = 40.0
u0 = [9990, 10, 0]
p = [0.05, 10.0, 0.25]
Random.seed!(1234)

des_model = MakeSIRModel(u0, p)
activate(des_model)
# Benchmark the simulation run
bench_result = @benchmark sir_run($des_model, $tmax) samples=10 evals=1

# Display benchmark results
println("\nBenchmark results:")
println("  Minimum time: $(round(minimum(bench_result).time / 1e9, digits=4)) seconds")
println("  Median time:  $(round(median(bench_result).time / 1e9, digits=4)) seconds")
println("  Mean time:    $(round(mean(bench_result).time / 1e9, digits=4)) seconds")
println("  Maximum time: $(round(maximum(bench_result).time / 1e9, digits=4)) seconds")
