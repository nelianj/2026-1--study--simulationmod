# # Оптимизация параметров модели SIR
#
# **Цель:** Найти оптимальные параметры модели, которые минимизируют
# пиковую заболеваемость и смертность от эпидемии.
#
# ## Подготовка окружения

using DrWatson
@quickactivate "project"

using BlackBoxOptim, Random, Statistics

include(srcdir("sir_model.jl"))

# ## Целевая функция для оптимизации
#
# Функция получает три параметра и возвращает взвешенную сумму
# пиковой заболеваемости и доли умерших.
# - **x[1]**: β_und - коэффициент заразности
# - **x[2]**: detection_time - время до выявления заболевания (дни)
# - **x[3]**: death_rate - вероятность летального исхода

function cost_multi(x)# x[1]: β_und, x[2]: detection_time, x[3]: death_rate    
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
   
    weight_peak = 0.7 # Объединяем обе цели в одну: взвешенная сумма
    weight_deaths = 0.3
    return weight_peak * mean(peak_vals) + weight_deaths * mean(dead_vals)
end

# ## Запуск оптимизации
#
# Используем эволюционный алгоритм для поиска оптимальных параметров.
# - **Метод:** adaptive_de_rand_1_bin (адаптивная дифференциальная эволюция)
# - **Время выполнения:** 60 секунд
# - **Размер популяции:** 20

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

# ## Результаты оптимизации

best = best_candidate(result)
fitness = best_fitness(result)

println("Оптимальные параметры:")
println("β_und = $(best[1])")
println("Время выявления = $(round(Int, best[2])) дней")
println("Смертность = $(best[3])")
println("Достигнутый показатель (взвешенная сумма): $(fitness)")

# ## Сохранение результатов

save(datadir("optimization_result.jld2"), Dict("best" => best, "fitness" => fitness))
