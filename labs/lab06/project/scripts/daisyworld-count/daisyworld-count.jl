using DrWatson
@quickactivate "project"
using Agents      # для агентного моделирования
using DataFrames  # для работы с данными
using Plots       # базовый бэкенд для графиков

include(srcdir("daisyworld.jl"))

using CairoMakie  # для высококачественной визуализации

black(a::Daisy) = a.breed == :black
white(a::Daisy) = a.breed == :white

adata = [(black, count), (white, count)]

model = daisyworld(; solar_luminosity = 1.0)

println("🔄 Запуск моделирования на 1000 шагов...")
agent_df, model_df = run!(model, 1000; adata)

figure = Figure(size = (600, 400))

ax = Axis(figure[1, 1],
          xlabel = "Время (шаги)",
          ylabel = "Количество маргариток",
          title = "Динамика численности маргариток в модели Daisyworld")

black_line = lines!(ax, agent_df[!, :time], agent_df[!, :count_black],
                    color = :black, linewidth = 2, label = "Чёрные")
white_line = lines!(ax, agent_df[!, :time], agent_df[!, :count_white],
                    color = :orange, linewidth = 2, label = "Белые")

Legend(figure[1, 2], [black_line, white_line], ["Чёрные", "Белые"], labelsize = 12)

display(figure)

save(plotsdir("daisy_count.png"), figure)
println("График сохранён: ", plotsdir("daisy_count.png"))
