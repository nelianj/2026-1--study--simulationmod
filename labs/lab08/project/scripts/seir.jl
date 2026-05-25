# Пример использования модели SEIR

using DrWatson
@quickactivate "project"
include(srcdir("Seir.jl"))
using Random, StatsPlots, BenchmarkTools, CSV, Dates
gr(fmt=:png)

# Параметры модели SEIR
tmax = 60.0
u0 = [990, 0, 10, 0]  # S, E, I, R
p = [0.05, 10.0, 0.25]  # β, c, γ
σ = 0.2  # Скорость перехода из E в I (1/средняя длительность латентного периода)

Random.seed!(1234)

# Запуск SEIR модели
seir_model = MakeSEIRModel(u0, p, σ)
activate_seir(seir_model)
seir_run(seir_model, tmax)
data_seir = out_seir(seir_model)

# Визуализация
plot1 = @df data_seir plot(
    :t,
    [:S :E :I :R],
    labels = ["S" "E" "I" "R"],
    xlab = "Время",
    ylab = "Численность",
    title = "Дискретно-событийная SEIR модель (σ = $σ)",
    linewidth = 2,
)
savefig(plotsdir("seir_des.png"))
display(plot1)

