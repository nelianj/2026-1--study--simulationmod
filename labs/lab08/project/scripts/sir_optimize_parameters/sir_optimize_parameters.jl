using DrWatson
@quickactivate "project"

using BlackBoxOptim, Random, Statistics

include(srcdir("sir_model.jl"))

function cost_multi(x)
    replicates = 3
    peak_vals = Float64[]
    dead_vals = Float64[]

    for rep = 1:replicates
        model = initialize_sir(;
            Ns = [1000, 1000, 1000],
            β_und = fill(x[1], 3),
            β_det = fill(x[1]/10, 3),
            infection_period = 14,
            detection_time = round(Int, x[2]),
            death_rate = x[3],
            reinfection_probability = 0.1,
            Is = [0, 0, 1],
            seed = 42 + rep,
            n_steps = 100,
        )

        infected_frac(model) = count(a.status == :I for a in allagents(model)) / nagents(model)

        peak_infected = 0.0

        for step = 1:100
            Agents.step!(model, 1)
            frac = infected_frac(model)
            if frac > peak_infected
                peak_infected = frac
            end
        end

        dead_count = 3000 - nagents(model)

        push!(peak_vals, peak_infected)
        push!(dead_vals, dead_count / 3000)
    end

    mean_peak = mean(peak_vals)
    mean_deaths = mean(dead_vals)

    if mean_peak > 0.30 # Штраф: если пик > 30%, возвращаем большое значение
        return 1000.0
    end

    return mean_deaths
end

result = bboptimize(
    cost_multi,
    Method = :adaptive_de_rand_1_bin,
    SearchRange = [
        (0.1, 1.0),   # β_und
        (3.0, 14.0),  # detection_time
        (0.01, 0.1),  # death_rate
    ],
    NumDimensions = 3,
    MaxTime = 60,
    PopulationSize = 20,
    TraceMode = :none,
)

best = best_candidate(result)
fitness = best_fitness(result)

println("Оптимальные параметры (при пике < 30%):")
println("β_und = $(round(best[1], digits=3))")
println("Время выявления = $(round(Int, best[2])) дней")
println("Смертность = $(round(best[3], digits=4))")
println("Доля умерших: $(round(fitness*100, digits=2))%")

save(datadir("optimization_result.jld2"), Dict("best" => best, "fitness" => fitness))
