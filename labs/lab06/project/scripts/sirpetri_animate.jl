# ## Анимация детерминированной динамики

using DrWatson
@quickactivate "project"

include(srcdir("SIRPetri.jl"))
using .SIRPetri
using Plots, Random

β = 0.3
γ = 0.1
tmax = 100.0
fps = 5

net, u0, names = build_sir_network(β, γ)
df = simulate_deterministic(net, u0, (0.0, tmax), saveat = 0.5, rates = [β, γ])
anim = @animate for row in eachrow(df)
    u = [row.S, row.I, row.R]
    bar(
        ["S", "I", "R"],
        u,
        legend = false,
        xlabel = "Compartments",
        ylabel = "Population",
        title = "t = $(round(row.time, digits=1))",
        ylims = (0, maximum(u0) + 5)
    )
end

gif(anim, plotsdir("sir_animation.gif"), fps = fps)

println("SIR animation saved to plots/sir_animation.gif")
