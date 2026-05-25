using DrWatson
@quickactivate "project"
include(srcdir("Sir_model.jl"))
using Random, StatsPlots, DataFrames, CSV

tmax = 40.0
u0 = [990, 10, 0]  # S, I, R
β_fixed = 0.05
γ_fixed = 0.25
Random.seed!(1234)

cs = [5, 10, 15, 20, 25, 30]

results_c = []

for c in cs
    p = [β_fixed, c, γ_fixed]
    model = MakeSIRModel(u0, p)
    activate(model)
    sir_run(model, tmax)
    data = out(model)

    peak_I = maximum(data.I)# Вычисляем ключевые метрики
    peak_time = data.t[argmax(data.I)]
    final_R = data.R[end]
    final_S = data.S[end]
    total_events = length(data.t)
    R0 = (c * β_fixed) / γ_fixed

    push!(results_c, (c=c, R0=R0, peak_I=peak_I, peak_time=peak_time,
                      final_S=final_S, final_R=final_R, events=total_events))

    println("c = $c (R₀ = $(round(R0, digits=2))): " *
            "Пик I = $(round(peak_I, digits=1)), " *
            "Время пика = $(round(peak_time, digits=1)), " *
            "Финальное R = $(round(final_R, digits=1))")
end

df_c = DataFrame(results_c)
CSV.write(datadir("sensitivity_c.csv"), df_c)

p1 = plot(df_c.c, [df_c.peak_I df_c.final_R],
          labels = ["Пик инфицированных" "Финальное число выздоровевших"],
          markershape = :circle, markersize = 6,
          xlabel = "c (частота контактов в день)",
          ylabel = "Численность",
          title = "Зависимость динамики эпидемии от частоты контактов",
          linewidth = 2,
          legend = :topleft)

savefig(plotsdir("sensitivity_c_peak.png"))
display(p1)

p2 = plot(df_c.c, df_c.peak_time,
          markershape = :circle, markersize = 6, color = :red,
          xlabel = "c (частота контактов в день)",
          ylabel = "Время достижения пика (дни)",
          title = "Зависимость времени пика от частоты контактов",
          linewidth = 2,
          label = "Время пика")

savefig(plotsdir("sensitivity_c_peak_time.png"))
display(p2)
