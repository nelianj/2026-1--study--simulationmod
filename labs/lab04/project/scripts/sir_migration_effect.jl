# # Исследование влияния миграции на динамику эпидемии
#
# **Цель:** Изучить, как интенсивность перемещения людей между городами влияет
# на скорость распространения эпидемии (время достижения пика) и масштаб пика.
# Инфекция начинается только в одном городе, остальные изначально здоровы.
#
# ## Подготовка окружения

using DrWatson
@quickactivate "project"
using Agents, DataFrames, Plots, CSV, Random, Statistics

include(srcdir("sir_model.jl"))

# ## Функция для создания матрицы миграции
#
# Создаёт матрицу вероятностей миграции между городами на основе заданной интенсивности.
# - **C**: количество городов
# - **intensity**: вероятность миграции в другой город
#
# Вероятность остаться в текущем городе: 1 - intensity
# Вероятность переехать в конкретный другой город: intensity / (C-1)

function create_migration_matrix(C, intensity)
    M = ones(C, C) .* intensity ./ (C-1)
    for i = 1:C
        M[i, i] = 1 - intensity
    end
    return M
end

# ## Функция для измерения времени достижения пика
#
# Запускает симуляцию с заданной интенсивностью миграции и возвращает:
# - **peak_time**: день, когда доля инфицированных достигла максимума
# - **peak_value**: максимальная доля инфицированных

function peak_time(p)
    migration_rates = create_migration_matrix(p[:C], p[:migration_intensity])# Создаём матрицу миграции на основе интенсивности
    
    model = initialize_sir(;
        Ns = p[:Ns],
        β_und = p[:β_und],
        β_det = p[:β_det],
        infection_period = p[:infection_period],
        detection_time = p[:detection_time],
        death_rate = p[:death_rate],
        reinfection_probability = p[:reinfection_probability],
        Is = p[:Is],
        seed = p[:seed],
        migration_rates = migration_rates,
    )
    
    infected_frac(model) = count(a.status == :I for a in allagents(model)) / nagents(model)
    peak = 0.0
    peak_step = 0
    
    for step = 1:p[:n_steps]
        agent_ids = collect(allids(model))
        for id in agent_ids
            agent = try
                model[id]
            catch
                nothing
            end
            if agent !== nothing
                sir_agent_step!(agent, model)
            end
        end
        
        frac = infected_frac(model)
        if frac > peak
            peak = frac
            peak_step = step
        end
    end
    
    return (peak_time = peak_step, peak_value = peak)
end# Ручной пошаговый запуск симуляции

# ## Параметры сканирования
#
# Исследуем интенсивность миграции от 0 (полная изоляция) до 0.5 (50% вероятность миграции)
# Для каждого значения выполняем 3 прогона с разными seed для учёта стохастичности.

migration_intensities = 0.0:0.1:0.5
seeds = [42, 43, 44]

# ## Создание списка параметров
#
# Формируем все комбинации интенсивности миграции и seed.

params_list = []
for mig in migration_intensities
    for s in seeds
        push!(
            params_list,
            Dict(
                :migration_intensity => mig,
                :C => 3,
                :Ns => [1000, 1000, 1000],
                :β_und => [0.5, 0.5, 0.5],
                :β_det => [0.05, 0.05, 0.05],
                :infection_period => 14,
                :detection_time => 7,
                :death_rate => 0.02,
                :reinfection_probability => 0.1,
                :Is => [1, 0, 0],  # инфекция начинается только в первом городе
                :seed => s,
                :n_steps => 150,
            ),
        )
    end
end

println("Всего экспериментов: ", length(params_list))

# ## Запуск экспериментов
#
# Последовательно запускаем все комбинации параметров.

results = []
for params in params_list
    data = peak_time(params)
    push!(results, merge(params, Dict(pairs(data))))
    println("Завершён эксперимент с migration_intensity = $(params[:migration_intensity]), seed = $(params[:seed]), пик = день $(data.peak_time), размер = $(round(data.peak_value * 100, digits=1))%")
end

# ## Сохранение результатов
#
# Сохраняем все прогоны в CSV-файл для дальнейшего анализа.

df = DataFrame(results)
CSV.write(datadir("migration_scan_all.csv"), df)
println("\nДанные сохранены в: $(datadir("migration_scan_all.csv"))")

# ## Усреднение по повторным прогонам
#
# Группируем результаты по интенсивности миграции и вычисляем средние показатели.
using Statistics
grouped = combine(
    groupby(df, [:migration_intensity]),
    :peak_time => mean => :mean_peak_time,
    :peak_value => mean => :mean_peak_value,
)

# Визуализация
plot(
    grouped.migration_intensity,
    grouped.mean_peak_time,
    marker = :circle,
    xlabel = "Интенсивность миграции",
    ylabel = "Время до пика (дни)",
    label = "Время пика",
)

plot!(
    grouped.migration_intensity,
    grouped.mean_peak_value .* 3000,
    marker = :square,
    xlabel = "Интенсивность миграции",
    ylabel = "Численность в пике",
    label = "Пиковая заболеваемость",
)
savefig(plotsdir("migration_effect.png"))

println("Результаты сохранены в data/migration_scan_all.csv и plots/migration_effect.png")
