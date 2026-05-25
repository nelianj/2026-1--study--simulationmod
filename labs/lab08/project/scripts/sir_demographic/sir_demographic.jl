using DrWatson
@quickactivate "project"

if !isdefined(Main, :SIRPerson)
    include(srcdir("Sir_model_1.jl"))
end

using StatsPlots
gr(fmt=:png)

p = [0.05, 10.0, 0.25, 0.01]
u0 = [990, 10, 0]
model = MakeSIRModel(u0, p)
activate(model)
sir_run(model, 100.0)

results = out(model)

plot(results.t, [results.S results.I results.R],
     label=["S" "I" "R"],
     xlabel="Время", ylabel="Численность популяции",
     title="SIR модель с демографией (μ=$(p[4]))")
savefig(plotsdir("sir_with_demography.png"))

println("Готово! μ = ", p[4])
