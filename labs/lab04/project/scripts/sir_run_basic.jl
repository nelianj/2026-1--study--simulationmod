# # Агентная модель SIR: Базовый эксперимент
#
# Надо Запустить базовый эксперимент с моделью SIR, проанализировать
# динамику эпидемии и сохранить результаты.
#
# Инициализируем проект DrWatson и подключаем необходимые пакеты.

using DrWatson
@quickactivate "project"
using Agents, DataFrames, Plots
gr()
default(show = :inline, fmt = :png)
using JLD2

# Подключаем модуль с определением модели SIR
include(srcdir("sir_model.jl"))

# ## Параметры эксперимента
#
# Определяем параметры модели:
# - **Ns**: численность населения в трёх городах
# - **β_und**: вероятность заражения невыявленными больными
# - **β_det**: вероятность заражения выявленными больными
# - **infection_period**: длительность заболевания
# - **detection_time**: время до выявления
# - **death_rate**: вероятность летального исхода
# - **reinfection_probability**: вероятность повторного заражения
# - **Is**: начальное количество инфицированных
# - **seed**: зерно генератора случайных чисел
# - **n_steps**: количество дней симуляции

params = Dict(
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

# Создаём модель с заданными параметрами.

model = initialize_sir(; params...)

# Создаём массивы для хранения динамики популяций на каждом шаге.

times = Int[]
S_vals = Int[]
I_vals = Int[]
R_vals = Int[]
total_vals = Int[]

# Выполняем пошаговую симуляцию на `n_steps` дней. На каждом шаге собираем статистику о состоянии популяции.

println("Запуск симуляции на $(params[:n_steps]) дней...")

for step = 1:params[:n_steps]
    Agents.step!(model, 1) # Выполняем один шаг моделирования
    
    push!(times, step)
    push!(S_vals, susceptible_count(model))
    push!(I_vals, infected_count(model))
    push!(R_vals, recovered_count(model))
    push!(total_vals, total_count(model))
    
    if step % 10 == 0 # Выводим прогресс каждые 10 шагов
        println("  День $step: I = $(I_vals[end]), R = $(R_vals[end])")
    end
end

println("Симуляция завершена.")

# Преобразуем собранные данные в удобный табличный формат.

agent_df = DataFrame(
    time = times,
    susceptible = S_vals,
    infected = I_vals,
    recovered = R_vals
)

model_df = DataFrame(
    time = times,
    total = total_vals
)

# Выводим итоговые показатели
println("Общая численность населения: $(total_vals[end])")
println("Переболело: $(R_vals[end]) человек ($(round(R_vals[end]/1000*100, digits=1))%)")
println("Умерло: $(total_vals[1] - total_vals[end]) человек")
println("Максимальное число инфицированных: $(maximum(I_vals))")

# ## Визуализация результатов
#
# Строим график динамики эпидемии:
# - Синяя линия: восприимчивые (S)
# - Красная линия: инфицированные (I)
# - Зелёная линия: выздоровевшие (R)
# - Пунктирная линия: общая численность (с учётом умерших)

plot(
    agent_df.time,
    agent_df.susceptible,
    label = "Восприимчивые (S)",
    xlabel = "Дни",
    ylabel = "Количество людей",
    title = "Динамика эпидемии (SIR модель)",
    linewidth = 2,
    color = :blue,
)
plot!(agent_df.time, agent_df.infected, label = "Инфицированные (I)", linewidth = 2, color = :red)
plot!(agent_df.time, agent_df.recovered, label = "Выздоровевшие (R)", linewidth = 2, color = :green)
plot!(agent_df.time, model_df.total, label = "Всего (включая умерших)", linestyle = :dash, linewidth = 1.5, color = :black)

# Сохраняем график
savefig(plotsdir("sir_basic_dynamics.png"))
println("\nГрафик сохранён")

# Сохраняем результаты в формате JLD2 для дальнейшего анализа.
@save datadir("sir_basic_agent.jld2") agent_df
@save datadir("sir_basic_model.jld2") model_df

println("Данные сохранены")
nothing
