# ## Модель SIR
#
# Модель создается, процессы активируются, и запуск выполняется. Результат преобразуется во фрейм данных. График с кривыми S, I, R создается и сохраняется

using DrWatson
@quickactivate "project"
include(srcdir("Sir_model.jl"))
using Random, StatsPlots, DataFrames, CSV

# **Разные параметры β**
# Фиксированные параметры
tmax = 40.0
u0 = [990, 10, 0] # S, I, R
c = 10.0
gamma = 0.25

betas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
results_beta = []

for beta in betas
    p = [beta, c, gamma]
    model = MakeSIRModel(u0, p)
    activate(model)
    sir_run(model, tmax)
    data = out(model)
    
    peak_I = maximum(data.I)  # Вычисляем ключевые метрики
    peak_time = data.t[argmax(data.I)]
    final_R = data.R[end]
    final_S = data.S[end]
    total_events = length(data.t)
    R0 = (c * beta) / gamma
    
    push!(results_beta, (beta=beta, R0=R0, peak_I=peak_I, peak_time=peak_time,final_S=final_S, final_R=final_R, events=total_events))
    
    println("β = $beta (R₀ = $(round(R0, digits=2))): " *
            "Пик I = $(round(peak_I, digits=1)), " *
            "Время пика = $(round(peak_time, digits=1)), " *
            "Финальное R = $(round(final_R, digits=1))")
end

# Сохраняем результаты
df_beta = DataFrame(results_beta)
CSV.write(datadir("sensitivity_beta.csv"), df_beta)

# График 1: Пик инфицированных и финальное R от β
p1 = plot(df_beta.beta, [df_beta.peak_I df_beta.final_R],
          labels = ["Пик инфицированных" "Финальное число выздоровевших"],
          markershape = :circle, markersize = 6,
          xlabel = "β (вероятность заражения)",
          ylabel = "Численность",
          title = "Зависимость динамики эпидемии от β",
          linewidth = 2,
          legend = :topleft)

savefig(plotsdir("sensitivity_beta_peak.png"))
display(p1)

# График 2: Время достижения пика от β
p2 = plot(df_beta.beta, df_beta.peak_time,
          markershape = :circle, markersize = 6, color = :red,
          xlabel = "β (вероятность заражения)",
          ylabel = "Время достижения пика (дни)",
          title = "Зависимость времени пика от β",
          linewidth = 2,
          label = "Время пика")

savefig(plotsdir("sensitivity_beta_peak_time.png"))
display(p2)
