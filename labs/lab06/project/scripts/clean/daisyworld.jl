using DrWatson
@quickactivate "project"

using Agents      # для агентного моделирования
using DataFrames  # для работы с данными
using Plots       # для визуализации (бэкенд)

include(srcdir("daisyworld.jl"))

using CairoMakie

model = daisyworld()

daisycolor(a::Daisy) = a.breed  # :black или :white

plotkwargs = (
    agent_color = daisycolor,
    agent_size = 20,
    agent_marker = '✿',
    heatarray = :temperature,
    heatkwargs = (colorrange = (-20, 60),),
)

println("Визуализация начального состояния (t = 0)")
plt1, _ = abmplot(model; plotkwargs...)
plt1

step!(model, 5)
println("Визуализация после 5 шагов")
plt2, _ = abmplot(model; heatarray = model.temperature, plotkwargs...)
plt2

step!(model, 35)
println("Визуализация после 40 шагов")
plt3, _ = abmplot(model; heatarray = model.temperature, plotkwargs...)
plt3

println("\nВсе графики сохранены в каталоге: ", plotsdir())
println("   - daisy_step001.png (начальное состояние)")
println("   - daisy_step005.png (после 5 шагов)")
println("   - daisy_step040.png (после 40 шагов)")

save(plotsdir("daisy_step001.png"), plt1)
save(plotsdir("daisy_step005.png"), plt2)
save(plotsdir("daisy_step040.png"), plt3)
