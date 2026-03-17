using DrWatson
@quickactivate "project"

using Agents      # для агентного моделирования
using DataFrames  # для работы с данными
using Plots       # для визуализации

include(srcdir("daisyworld.jl"))

model = daisyworld()

function plot_daisyworld(model, title_str)
    p1 = heatmap(model.temperature',
                title = "Температура поверхности\n$title_str",
                xlabel = "X", ylabel = "Y",
                color = :thermal, clims = (-20, 60))

    black_x, black_y, white_x, white_y = Int[], Int[], Int[], Int[]
    for agent in allagents(model)
        if agent.breed == :black
            push!(black_x, agent.pos[1]); push!(black_y, agent.pos[2])
        else
            push!(white_x, agent.pos[1]); push!(white_y, agent.pos[2])
        end
    end

    !isempty(black_x) && scatter!(p1, black_x, black_y, color=:black, marker=:star5, label="Чёрные")
    !isempty(white_x) && scatter!(p1, white_x, white_y, color=:white, marker=:star5, label="Белые")

    return p1
end

plt1 = plot_daisyworld(model, "t = 0")
savefig(plt1, plotsdir("daisy_step001.png"))

step!(model, 5)
plt2 = plot_daisyworld(model, "t = 5")
savefig(plt2, plotsdir("daisy_step005.png"))

step!(model, 35)
plt3 = plot_daisyworld(model, "t = 40")
savefig(plt3, plotsdir("daisy_step040.png"))
