using DrWatson
@quickactivate "project"
using Agents, DataFrames, Plots
using JLD2

include(srcdir("sir_model.jl"))

base_params = Dict(
    :Ns => [1000, 1000, 1000],
    :β_und => [0.5, 0.5, 0.5],
    :β_det => [0.05, 0.05, 0.05],
    :infection_period => 14,
    :detection_time => 7,
    :death_rate => 0.02,
    :reinfection_probability => 0.1,
    :Is => [0, 0, 1],
    :seed => 42,
    :n_steps => 100,
)

param_grid = Dict(
    :β_und => [[0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9]],
    :death_rate => [0.01, 0.02, 0.05, 0.1],
    :detection_time => [3, 7, 10, 14],
)

grid_params = dict_list(param_grid)

all_params = [merge(base_params, p) for p in grid_params]

println("Всего комбинаций параметров: ", length(all_params))

function run_experiment(p)
    model = initialize_sir(;
        Ns = p[:Ns],
        β_und = p[:β_und],
        β_det = [x/10 for x in p[:β_und]],  # β_det = β_und/10
        infection_period = p[:infection_period],
        detection_time = p[:detection_time],
        death_rate = p[:death_rate],
        reinfection_probability = p[:reinfection_probability],
        Is = p[:Is],
        seed = p[:seed],
    )

    times = Int[]
    I_vals = Int[]
    R_vals = Int[]
    total_vals = Int[]

    for step = 1:p[:n_steps]
        Agents.step!(model, 1)

        push!(times, step)
        push!(I_vals, infected_count(model))
        push!(R_vals, recovered_count(model))
        push!(total_vals, total_count(model))
    end

    return Dict(
        :peak_infected => maximum(I_vals),
        :peak_time => times[argmax(I_vals)],
        :final_recovered => R_vals[end],
        :final_total => total_vals[end],
        :total_deaths => total_vals[1] - total_vals[end],
        :infected_ratio => maximum(I_vals) / total_vals[1] * 100,
    )
end

results = []

for (i, p) in enumerate(all_params)
    println("Эксперимент $i/$(length(all_params))")
    println("  β_und = $(p[:β_und][1]), death_rate = $(p[:death_rate]), detection_time = $(p[:detection_time])")

    res = run_experiment(p)

    push!(results, merge(p, res))

    println("  Пик: $(res[:peak_infected]) чел. ($(round(res[:infected_ratio], digits=1))%)")
    println("  Умерло: $(res[:total_deaths]) чел.")
    println()
end

results_df = DataFrame(results)

println("\nТАБЛИЦА РЕЗУЛЬТАТОВ")
println(results_df[:, [:β_und, :death_rate, :detection_time, :peak_infected, :total_deaths]])

@save datadir("sir_parameter_sweep_results.jld2") results_df

p1 = plot(
    [p[:β_und][1] for p in results],
    [p[:peak_infected] for p in results],
    seriestype = :scatter,
    label = "Пик инфицированных",
    xlabel = "Коэффициент заразности β",
    ylabel = "Максимальное число инфицированных",
    title = "Влияние заразности на пик эпидемии",
    markersize = 8,
    grid = true,
)

savefig(plotsdir("sir_beta_effect.png"))
println("\nГрафик 1 сохранён: $(plotsdir("sir_beta_effect.png"))")

p2 = plot(
    [p[:death_rate] for p in results],
    [p[:total_deaths] for p in results],
    seriestype = :scatter,
    label = "Общее число умерших",
    xlabel = "Вероятность смерти",
    ylabel = "Количество умерших",
    title = "Влияние смертности на число жертв",
    markersize = 8,
    grid = true,
)

savefig(plotsdir("sir_death_effect.png"))
println("График 2 сохранён: $(plotsdir("sir_death_effect.png"))")

p3 = plot(
    [p[:detection_time] for p in results],
    [p[:peak_infected] for p in results],
    seriestype = :scatter,
    label = "Пик инфицированных",
    xlabel = "Время выявления (дни)",
    ylabel = "Максимальное число инфицированных",
    title = "Влияние скорости выявления на пик эпидемии",
    markersize = 8,
    grid = true,
)

savefig(plotsdir("sir_detection_effect.png"))
println("График 3 сохранён: $(plotsdir("sir_detection_effect.png"))")

best_scenario = results[argmin([p[:total_deaths] for p in results])]
worst_scenario = results[argmax([p[:total_deaths] for p in results])]

println("\nАНАЛИЗ СЦЕНАРИЕВ")
println("Лучший сценарий (минимальная смертность):")
println("  β = $(best_scenario[:β_und][1]), смертность = $(best_scenario[:death_rate])")
println("  время выявления = $(best_scenario[:detection_time]) дней")
println("  пик = $(best_scenario[:peak_infected]) чел., умерло = $(best_scenario[:total_deaths]) чел.")
println()
println("Худший сценарий (максимальная смертность):")
println("  β = $(worst_scenario[:β_und][1]), смертность = $(worst_scenario[:death_rate])")
println("  время выявления = $(worst_scenario[:detection_time]) дней")
println("  пик = $(worst_scenario[:peak_infected]) чел., умерло = $(worst_scenario[:total_deaths]) чел.")
