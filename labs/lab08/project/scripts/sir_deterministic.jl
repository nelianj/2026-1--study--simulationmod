# ## Модель SIR
#
# Модель создается, процессы активируются, и запуск выполняется. Результат преобразуется во фрейм данных. График с кривыми S, I, R создается и сохраняется

using DrWatson
@quickactivate "project"
include(srcdir("Sir_model.jl"))
include(srcdir("Sir_deterministic.jl"))
using Random, StatsPlots, BenchmarkTools
gr(fmt=:png)

# Параметры модели
tmax = 40.0
u0 = [990, 10, 0] # S, I, R
p = [0.05, 10.0, 0.25] # β, c, γ

Random.seed!(1234)

# Запуск модели
stoch_model = MakeSIRModelD(u0, p)
activate(stoch_model)
sir_run(stoch_model, tmax)
data_stoch = out(stoch_model)

deter_model = MakeSIRModelD(u0, p)
activate(deter_model)
sir_run(deter_model, tmax)
data_deter = out(deter_model)

# Визуализация
plot1 = @df data_deter plot(
    :t,
    [:S :I :R],
    labels = ["S" "I" "R"],
    xlab = "Время",
    ylab = "Численность",
    title = "Детерминированная SIR модель (фиксированное выздоровление)",
)
savefig(plotsdir("sir_deter.png"))
display(plot1)

plot2 = @df data_stoch plot(
    :t,
    [:S :I :R],
    labels = ["S" "I" "R"],
    xlab = "Время",
    ylab = "Численность",
    title = "Стохастическая SIR модель (экспоненциальное выздоровление)",
)
savefig(plotsdir("sir_stoch.png"))
display(plot2)

plot3 = plot(
    data_stoch.t, data_stoch.I,
    label = "Стохастическая (экспоненциальная)",
    xlabel = "Время",
    ylabel = "Численность инфицированных (I)",
    title = "Сравнение моделей SIR: динамика инфицированных",
    linewidth = 2,
    color = :red,
    legend = :topright,
    grid = true,
)
plot!(
    plot3,
    data_deter.t, data_deter.I,
    label = "Детерминированная (фиксированная)",
    linewidth = 2,
    color = :blue,
)
savefig(plotsdir("sir_comparison.png"))
