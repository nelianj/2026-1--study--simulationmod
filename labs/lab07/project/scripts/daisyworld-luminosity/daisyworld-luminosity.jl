using DrWatson
@quickactivate "project"
using Agents      # для агентного моделирования
using DataFrames  # для работы с данными
using Plots       # базовый бэкенд для графиков
using CairoMakie  # для высококачественной визуализации

include(srcdir("daisyworld.jl"))

using CairoMakie  # для высококачественной визуализации

black(a::Daisy) = a.breed == :black
white(a::Daisy) = a.breed == :white

adata = [(black, count), (white, count)]  # данные об агентах

temperature(model) = StatsBase.mean(model.temperature)
mdata = [temperature, :solar_luminosity]  # данные о модели

println("Создание модели Daisyworld со сценарием :ramp...")
model = daisyworld(; solar_luminosity = 1.0, scenario = :ramp)

println("Запуск моделирования на 1000 шагов...")
agent_df, model_df = run!(model, 1000; adata = adata, mdata = mdata)

println("Моделирование завершено")
println("   - Собрано $(nrow(agent_df)) записей о популяциях")
println("   - Собрано $(nrow(model_df)) записей о состоянии модели")

figure = CairoMakie.Figure(size = (600, 600))

ax1 = Axis(figure[1, 1],
           ylabel = "Количество маргариток",
           title = "Динамика популяций")

black_line = lines!(ax1, agent_df[!, :time], agent_df[!, :count_black],
                    color = :red, linewidth = 2, label = "Чёрные")
white_line = lines!(ax1, agent_df[!, :time], agent_df[!, :count_white],
                    color = :blue, linewidth = 2, label = "Белые")

figure[1, 2] = Legend(figure, [black_line, white_line],
                      ["Чёрные", "Белые"], labelsize = 12)

ax2 = Axis(figure[2, 1],
           ylabel = "Средняя температура (°C)")

lines!(ax2, model_df[!, :time], model_df[!, :temperature],
       color = :red, linewidth = 2)

ax3 = Axis(figure[3, 1],
           xlabel = "Время (шаги)",
           ylabel = "Солнечная светимость")

lines!(ax3, model_df[!, :time], model_df[!, :solar_luminosity],
       color = :red, linewidth = 2)

for ax in (ax1, ax2)
    ax.xticklabelsvisible = false
    ax.xlabel = ""
end

display(figure)

save(plotsdir("daisy_luminosity.png"), figure)
println("График сохранён: ", plotsdir("daisy_luminosity.png"))
