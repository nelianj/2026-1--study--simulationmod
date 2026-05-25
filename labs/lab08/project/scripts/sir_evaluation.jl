using DrWatson
@quickactivate "project"
include(srcdir("Sir_model.jl"))
using Random, StatsPlots, BenchmarkTools

# Параметры модели
tmax = 40.0
u0 = [99990, 10, 0] # S, I, R
p = [0.05, 10.0, 0.25] # β, c, γ

Random.seed!(1234)

# Запуск модели
des_model = MakeSIRModel(u0, p)
activate(des_model)

bench_result = @benchmark sir_run($des_model, $tmax) samples = 10 evals = 1

# Результат
println("\nРезультаты бенчмаркинга для N=10 000:")
println("  Минимальное время: $(round(minimum(bench_result).time / 1e9, digits=4)) секунд")
println("  Медианное время:  $(round(median(bench_result).time / 1e9, digits=4)) секунд")
println("  Среднее время:    $(round(mean(bench_result).time / 1e9, digits=4)) секунд")
println("  Максимальное время: $(round(maximum(bench_result).time / 1e9, digits=4)) секунд")
