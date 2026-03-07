# # Модель Лотки-Вольтерры (Хищник-Жертва)
# 
# Этот скрипт реализует классическую модель Лотки-Вольтерры для описания
# динамики взаимодействия двух видов: хищников и жертв.
# 
# **Версия с параметрическим сканированием**

# ## 1. Инициализация проекта и загрузка пакетов

using DrWatson
@quickactivate "project"

using DifferentialEquations  # Решение ОДУ
using DataFrames             # Работа с данными
using Plots                   # Визуализация
using LaTeXStrings            # Красивые формулы
using Statistics              # Статистический анализ
using FFTW                    # Спектральный анализ
using JLD2                    # Для сохранения данных
using CSV                     # Для сохранения в CSV

# Создание директорий для результатов
script_name = splitext(basename(PROGRAM_FILE))[1]
mkpath(plotsdir(script_name))
mkpath(datadir(script_name))

# ## 2. Математическое описание модели
# 
# Модель Лотки-Вольтерры описывается системой дифференциальных уравнений:
# 
# ```math
# \begin{cases}
# \frac{dx}{dt} = \alpha x - \beta xy \\
# \frac{dy}{dt} = \delta xy - \gamma y
# \end{cases}
# ```
# 
# где:
# - **x** - популяция жертв (например, зайцы)
# - **y** - популяция хищников (например, лисы)
# - **α** - естественный прирост жертв (в отсутствие хищников)
# - **β** - коэффициент поедания жертв хищниками
# - **δ** - коэффициент прироста хищников за счет поедания жертв
# - **γ** - естественная смертность хищников (в отсутствие жертв)

function lotka_volterra!(du, u, p, t)
    x, y = u 
    α, β, δ, γ = p

    @inbounds begin
        du[1] = α*x - β*x*y  # изменение популяции жертв
        du[2] = δ*x*y - γ*y   # изменение популяции хищников
    end
    nothing
end

# ## 3. Базовые параметры модели (один эксперимент)
# 
# Используем классические параметры из литературы:
# - **α = 0.1** - скорость размножения жертв
# - **β = 0.02** - скорость поедания жертв
# - **δ = 0.01** - коэффициент конверсии пищи в потомство
# - **γ = 0.3** - смертность хищников
# 
# Начальные условия:
# - **x₀ = 40** - начальная популяция жертв
# - **y₀ = 9** - начальная популяция хищников
# 
# Время моделирования: **200 единиц**

p_lv = [0.1, 0.02, 0.01, 0.3]
u0_lv = [40.0, 9.0]
tspan_lv = (0.0, 200.0) 
dt_lv = 0.01

# ## 4. Численное решение системы (базовый эксперимент)

prob_lv = ODEProblem(lotka_volterra!, u0_lv, tspan_lv, p_lv)
sol_lv = solve(prob_lv,
    dt = dt_lv,
    Tsit5(), 
    reltol=1e-8,
    abstol=1e-10,
    saveat=0.1,
    dense=true
)

# ## 5. Подготовка данных для анализа

df_lv = DataFrame()
df_lv[!, :t] = sol_lv.t
df_lv[!, :prey] = [u[1] for u in sol_lv.u]
df_lv[!, :predator] = [u[2] for u in sol_lv.u]

# Расчет производных для анализа скоростей изменения
df_lv[!, :dprey_dt] = p_lv[1] .* df_lv.prey .- p_lv[2] .* df_lv.prey .* df_lv.predator
df_lv[!, :dpredator_dt] = p_lv[3] .* df_lv.prey .* df_lv.predator .- p_lv[4] .* df_lv.predator

# Расчет относительных изменений (в процентах)
df_lv[!, :prey_pct_change] = df_lv.dprey_dt ./ df_lv.prey .* 100
df_lv[!, :predator_pct_change] = df_lv.dpredator_dt ./ df_lv.predator .* 100

# ## 6. Вывод информации о модели

println("="^60)
println("МОДЕЛЬ ЛОТКИ-ВОЛЬТЕРРЫ (ХИЩНИК-ЖЕРТВА)")
println("="^60)
println("\nПараметры модели:")
println("  α (скорость размножения жертв) = ", p_lv[1])
println("  β (скорость поедания жертв) = ", p_lv[2])
println("  δ (коэффициент конверсии) = ", p_lv[3])
println("  γ (смертность хищников) = ", p_lv[4])
println("\nНачальные условия:")
println("  Жертвы (x₀) = ", u0_lv[1])
println("  Хищники (y₀) = ", u0_lv[2])

# ## 7. Стационарные точки (положения равновесия)
# 
# ```math
# x^* = \frac{\gamma}{\delta}, \quad y^* = \frac{\alpha}{\beta}
# ```

x_star = p_lv[4] / p_lv[3]
y_star = p_lv[1] / p_lv[2]

println("\nСтационарные точки (положения равновесия):")
println("  x* = γ/δ = ", round(x_star, digits=3))
println("  y* = α/β = ", round(y_star, digits=3))

# ============================================================
# 8. ВИЗУАЛИЗАЦИЯ БАЗОВОГО ЭКСПЕРИМЕНТА
# ============================================================

# ## 8.1 График 1: Динамика популяций во времени
# 
# Отображает изменение численности жертв и хищников с течением времени.

plt1 = plot(df_lv.t, [df_lv.prey df_lv.predator],
    label=[L"Жертвы (x)" L"Хищники (y)"],
    xlabel="Время",
    ylabel="Популяция",
    title="Модель Лотки-Вольтерры: Динамика популяций",
    linewidth=2,
    legend=:topright,
    grid=true,
    size=(900, 500),
    color=[:green :red])

# Добавление стационарных уровней
hline!(plt1, [x_star], color=:green, linestyle=:dash, alpha=0.5, 
       label="x* (равновесие жертв)")
hline!(plt1, [y_star], color=:red, linestyle=:dash, alpha=0.5, 
       label="y* (равновесие хищников)")

# ## 8.2 График 2: Фазовый портрет (хищники vs жертвы)
# 
# Показывает зависимость численности хищников от численности жертв.
# Замкнутые траектории указывают на циклический характер системы.

plt2 = plot(df_lv.prey, df_lv.predator,
    label="Фазовая траектория",
    xlabel="Популяция жертв (x)",
    ylabel="Популяция хищников (y)",
    title="Фазовый портрет системы",
    color=:blue,
    linewidth=1.5,
    grid=true,
    size=(800, 600),
    legend=:topright)

# Добавление стрелок направления на фазовом портрете
step = 50 
for i in 1:step:length(df_lv.prey)-step
    plot!(plt2, 
        [df_lv.prey[i], df_lv.prey[i+step]],
        [df_lv.predator[i], df_lv.predator[i+step]],
        arrow=:closed, color=:blue, alpha=0.3, label=false)
end

# Добавление стационарной точки
scatter!(plt2, [x_star], [y_star],
    color=:black, markersize=8, label="Стационарная точка (x*, y*)")

# Изоклины (линии нулевого роста)
x_range = LinRange(0, maximum(df_lv.prey)*1.1, 100)
y_nullcline = p_lv[1] ./ (p_lv[2] .* x_range)  # dy/dt = 0
plot!(plt2, x_range, y_nullcline,
    color=:red, linestyle=:dash, linewidth=1.5, label="Изоклина хищников (dy/dt=0)")

y_range = LinRange(0, maximum(df_lv.predator)*1.1, 100)
x_nullcline = p_lv[4] ./ (p_lv[3] .* ones(length(y_range)))  # dx/dt = 0
plot!(plt2, x_nullcline, y_range,
    color=:green, linestyle=:dash, linewidth=1.5, label="Изоклина жертв (dx/dt=0)")

# ## 8.3 График 3: Производные (скорости изменения)

plt3 = plot(df_lv.t, [df_lv.dprey_dt df_lv.dpredator_dt],
    label=[L"dx/dt" L"dy/dt"],
    xlabel="Время",
    ylabel="Скорость изменения",
    title="Производные популяций",
    linewidth=1.5,
    legend=:topright,
    grid=true,
    size=(900, 400),
    color=[:green :red])

hline!(plt3, [0], color=:black, linestyle=:solid, alpha=0.3, label=false)

# ## 8.4 График 4: Относительные изменения (в процентах)

plt4 = plot(df_lv.t, [df_lv.prey_pct_change df_lv.predator_pct_change],
    label=[L"dx/dt / x (\%)" L"dy/dt / y (\%)"],
    xlabel="Время",
    ylabel="Относительное изменение, %",
    title="Относительные темпы роста",
    linewidth=1.5,
    legend=:topright,
    grid=true,
    size=(900, 400),
    color=[:green :red])

# ## 8.5 График 5: Спектральный анализ (FFT)
# 
# Функция для вычисления быстрого преобразования Фурье

function compute_fft(signal, dt)
    n = length(signal)
    spectrum = abs.(rfft(signal .- mean(signal)))
    freq = rfftfreq(n, 1/dt)
    return freq, spectrum
end

# Вычисление периодов колебаний
freq_prey, spectrum_prey = compute_fft(df_lv.prey, dt_lv)
freq_predator, spectrum_predator = compute_fft(df_lv.predator, dt_lv)

plt5 = plot(freq_prey, [spectrum_prey spectrum_predator],
    label=[L"Жертвы (x)" L"Хищники (y)"],
    xlabel="Частота",
    ylabel="Амплитуда",
    title="Спектральный анализ (Фурье)",
    linewidth=1.5,
    xscale=:log10,
    yscale=:log10,
    legend=:topright,
    grid=true,
    size=(800, 400),
    color=[:green :red])

# Нахождение доминирующих частот
if length(spectrum_prey) > 0
    idx_prey = argmax(spectrum_prey[2:end]) + 1  # пропускаем нулевую частоту
    dominant_freq_prey = freq_prey[idx_prey]
    period_prey = 1/dominant_freq_prey
    
    println("\nСпектральный анализ:")
    println("  Доминирующая частота колебаний жертв: ", round(dominant_freq_prey, digits=4))
    println("  Период колебаний жертв: ", round(period_prey, digits=2), " ед. времени")
end

# ## 8.6 График 6: Компактная панель всех графиков

plt6 = plot(layout=(3, 2), size=(1200, 900))

plot!(plt6[1], df_lv.t, df_lv.prey, label=L"x(t)", color=:green, linewidth=2,
    title="Популяция жертв", grid=true)
plot!(plt6[2], df_lv.t, df_lv.predator, label=L"y(t)", color=:red, linewidth=2,
    title="Популяция хищников", grid=true)
plot!(plt6[3], df_lv.prey, df_lv.predator, label=false, color=:blue, linewidth=1.5,
    title="Фазовый портрет", xlabel=L"x", ylabel=L"y", grid=true)
scatter!(plt6[3], [x_star], [y_star], color=:black, markersize=5, label="(x*, y*)")
plot!(plt6[4], df_lv.t, [df_lv.dprey_dt df_lv.dpredator_dt],
    label=[L"dx/dt" L"dy/dt"], color=[:green :red], linewidth=1.5,
    title="Скорости изменения", grid=true, legend=:topright)
plot!(plt6[5], freq_prey, spectrum_prey, label=L"x", color=:green, linewidth=1.5,
    title="Спектр жертв", xscale=:log10, yscale=:log10, grid=true)
plot!(plt6[6], df_lv.t, [df_lv.prey_pct_change df_lv.predator_pct_change],
    label=[L"dx/x" L"dy/y"], color=[:green :red], linewidth=1.5,
    title="Относительные изменения", grid=true, legend=:topright)

# ============================================================
# 9. АНАЛИЗ БАЗОВОГО ЭКСПЕРИМЕНТА
# ============================================================

println("\n" * "="^60)
println("АНАЛИЗ БАЗОВОГО ЭКСПЕРИМЕНТА")
println("="^60)

println("\nОсновные статистики:")
println("  Жертвы: min = $(round(minimum(df_lv.prey), digits=2)), " *
        "max = $(round(maximum(df_lv.prey), digits=2)), " *
        "mean = $(round(mean(df_lv.prey), digits=2))")
println("  Хищники: min = $(round(minimum(df_lv.predator), digits=2)), " *
        "max = $(round(maximum(df_lv.predator), digits=2)), " *
        "mean = $(round(mean(df_lv.predator), digits=2))")

# ## 9.1 Анализ колебаний
# 
# Функция для поиска первого пика в сигнале

function find_first_peak(signal, time)
    for i in 2:length(signal)-1
        if signal[i] > signal[i-1] && signal[i] > signal[i+1]
            return time[i], signal[i]
        end
    end
    return NaN, NaN
end

peak_time_prey, peak_value_prey = find_first_peak(df_lv.prey, df_lv.t)
peak_time_predator, peak_value_predator = find_first_peak(df_lv.predator, df_lv.t)

if !isnan(peak_time_prey) && !isnan(peak_time_predator)
    phase_shift = peak_time_predator - peak_time_prey
    println("\nАнализ колебаний:")
    println("  Первый пик жертв: время = $(round(peak_time_prey, digits=2)), " *
            "значение = $(round(peak_value_prey, digits=2))")
    println("  Первый пик хищников: время = $(round(peak_time_predator, digits=2)), " *
            "значение = $(round(peak_value_predator, digits=2))")
    println("  Сдвиг фаз (хищники отстают): $(round(phase_shift, digits=2)) ед. времени")
end

# ============================================================
# 10. ПАРАМЕТРИЧЕСКОЕ СКАНИРОВАНИЕ
# ============================================================
# 
# ## 10.1 Зачем нужно параметрическое сканирование?
# 
# До сих пор мы рассматривали только один набор параметров. 
# Но как изменится динамика системы, если:
# - **α** (скорость размножения жертв) будет другой?
# - **β** (скорость поедания) изменится?
# - **δ** (эффективность конверсии) будет иной?
# - **γ** (смертность хищников) поменяется?
# 
# Чтобы ответить на эти вопросы, проведём **параметрическое сканирование** -
# запустим модель с разными комбинациями параметров и сравним результаты.

println("\n" * "="^60)
println("ПАРАМЕТРИЧЕСКОЕ СКАНИРОВАНИЕ")
println("="^60)

# ## 10.2 Сетка параметров
# 
# Создадим словарь с наборами значений для каждого параметра.
# Мы будем варьировать 4 параметра:
# - **α**: скорость размножения жертв
# - **β**: скорость поедания жертв
# - **δ**: коэффициент конверсии
# - **γ**: смертность хищников

param_grid = Dict(
    :α => [0.05, 0.1, 0.15, 0.2],        # скорость размножения жертв
    :β => [0.01, 0.02, 0.03, 0.04],       # скорость поедания жертв
    :δ => [0.005, 0.01, 0.015, 0.02],     # коэффициент конверсии
    :γ => [0.2, 0.3, 0.4, 0.5],           # смертность хищников
    :u0 => [[40.0, 9.0]],                  # начальные условия (фиксированы)
    :tspan => [(0.0, 200.0)]               # время моделирования (фиксировано)
)

# Создаём все возможные комбинации параметров вручную
# Всего комбинаций: 4 × 4 × 4 × 4 = 256 экспериментов

println("\nСоздание комбинаций параметров...")

all_params = []
for α in param_grid[:α]
    for β in param_grid[:β]
        for δ in param_grid[:δ]
            for γ in param_grid[:γ]
                push!(all_params, Dict(
                    :α => α,
                    :β => β,
                    :δ => δ,
                    :γ => γ,
                    :u0 => param_grid[:u0][1],
                    :tspan => param_grid[:tspan][1]
                ))
            end
        end
    end
end

println("Всего комбинаций параметров: ", length(all_params))  # 4×4×4×4 = 256
println("\nИсследуемые значения:")
println("  α: ", param_grid[:α])
println("  β: ", param_grid[:β])
println("  δ: ", param_grid[:δ])
println("  γ: ", param_grid[:γ])

# ## 10.3 Функция для запуска одного эксперимента
# 
# Эта функция принимает набор параметров и возвращает результаты.
# Она будет вызываться для каждой комбинации параметров.

function run_lv_experiment(params::Dict)
    α = params[:α]# Извлекаем параметры из словаря
    β = params[:β]
    δ = params[:δ]
    γ = params[:γ]
    u0 = params[:u0]
    tspan = params[:tspan]
    
    p = [α, β, δ, γ]
    prob = ODEProblem(lotka_volterra!, u0, tspan, p) # Создаём и решаем задачу
    sol = solve(prob, Tsit5(), saveat=1.0)  # сохраняем реже для экономии памяти
    df_temp = DataFrame(t=sol.t) # Анализируем результаты
    df_temp.prey = [u[1] for u in sol.u]
    df_temp.predator = [u[2] for u in sol.u]
    x_star_val = γ / δ # Стационарные точки
    y_star_val = α / β
    
    mean_prey = mean(df_temp.prey) # Статистика
    mean_predator = mean(df_temp.predator)
    min_prey = minimum(df_temp.prey)
    max_prey = maximum(df_temp.prey)
    min_predator = minimum(df_temp.predator)
    max_predator = maximum(df_temp.predator)
    
    prey_amplitude = (max_prey - min_prey) / 2 # Амплитуда колебаний
    predator_amplitude = (max_predator - min_predator) / 2
    
    return Dict(
        "solution" => sol,
        "mean_prey" => mean_prey,
        "mean_predator" => mean_predator,
        "min_prey" => min_prey,
        "max_prey" => max_prey,
        "min_predator" => min_predator,
        "max_predator" => max_predator,
        "prey_amplitude" => prey_amplitude,
        "predator_amplitude" => predator_amplitude,
        "x_star" => x_star_val,
        "y_star" => y_star_val
    )
end

# ## 10.4 Запуск всех экспериментов
# 
# Используем функцию `produce_or_load` для кэширования результатов.
# Это позволяет не пересчитывать эксперименты при повторных запусках.

all_results = []

for (i, params) in enumerate(all_params)
    println("\nПрогресс: $i/$(length(all_params))")
    println("  α = $(params[:α]), β = $(params[:β]), δ = $(params[:δ]), γ = $(params[:γ])")
    
    data, path = produce_or_load( # Запускаем эксперимент (или загружаем из кэша)
        datadir(script_name, "parametric_scan"),
        params,
        run_lv_experiment,
        prefix = "lv_scan",
        tag = false,
        verbose = false
    )
    
    result_summary = merge(params, Dict( # Сохраняем сводку результатов
        :mean_prey => data["mean_prey"],
        :mean_predator => data["mean_predator"],
        :prey_amplitude => data["prey_amplitude"],
        :predator_amplitude => data["predator_amplitude"],
        :x_star => data["x_star"],
        :y_star => data["y_star"]
    ))
    
    push!(all_results, result_summary)
end

# ## 10.5 Сводная таблица результатов

results_df = DataFrame(all_results)
println("\n" * "="^60)
println("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
println("="^60)
println(first(results_df, 10))  # показываем первые 10 строк

# ============================================================
# 11. ВИЗУАЛИЗАЦИЯ ПАРАМЕТРИЧЕСКОГО СКАНИРОВАНИЯ
# ============================================================

# ## 11.1 Зависимость средних значений от параметров
# 
# Исследуем, как параметры влияют на средние численности популяций.

# График: зависимость средних значений от α
p_alpha = scatter(results_df.α, [results_df.mean_prey results_df.mean_predator],
    label=[L"\bar{x}" L"\bar{y}"],
    xlabel=L"\alpha (скорость размножения жертв)",
    ylabel="Средняя численность",
    title="Зависимость средних значений от α",
    color=[:green :red],
    markersize=4,
    alpha=0.7,
    grid=true)

savefig(p_alpha, plotsdir(script_name, "param_scan_alpha.png"))

# График: зависимость средних значений от β
p_beta = scatter(results_df.β, [results_df.mean_prey results_df.mean_predator],
    label=[L"\bar{x}" L"\bar{y}"],
    xlabel=L"\beta (скорость поедания)",
    ylabel="Средняя численность",
    title="Зависимость средних значений от β",
    color=[:green :red],
    markersize=4,
    alpha=0.7,
    grid=true)

savefig(p_beta, plotsdir(script_name, "param_scan_beta.png"))

# ## 11.2 Зависимость амплитуды колебаний от параметров

p_amplitude = scatter(results_df.α, [results_df.prey_amplitude results_df.predator_amplitude],
    label=[L"A_x" L"A_y"],
    xlabel=L"\alpha",
    ylabel="Амплитуда колебаний",
    title="Зависимость амплитуды от α",
    color=[:green :red],
    markersize=4,
    alpha=0.7,
    grid=true)

savefig(p_amplitude, plotsdir(script_name, "param_scan_amplitude.png"))

# ## 11.3 Сравнение с теорией: стационарные точки
# 
# Теоретические стационарные точки:
# ```math
# x^* = \frac{\gamma}{\delta}, \quad y^* = \frac{\alpha}{\beta}
# ```

p_theory_x = scatter(results_df.γ ./ results_df.δ, results_df.mean_prey,
    xlabel=L"\gamma/\delta (теоретическое x^*)",
    ylabel=L"Средняя численность жертв \bar{x}",
    title="Сравнение с теорией: жертвы",
    color=:green,
    markersize=4,
    alpha=0.7,
    grid=true)

# Линия y = x для сравнения
min_val = min(minimum(results_df.γ ./ results_df.δ), minimum(results_df.mean_prey))
max_val = max(maximum(results_df.γ ./ results_df.δ), maximum(results_df.mean_prey))
plot!(p_theory_x, [min_val, max_val], [min_val, max_val],
    color=:red, linestyle=:dash, label="Теория: x* = γ/δ")

savefig(p_theory_x, plotsdir(script_name, "param_scan_theory_x.png"))

p_theory_y = scatter(results_df.α ./ results_df.β, results_df.mean_predator,
    xlabel=L"\alpha/\beta (теоретическое y^*)",
    ylabel=L"Средняя численность хищников \bar{y}",
    title="Сравнение с теорией: хищники",
    color=:red,
    markersize=4,
    alpha=0.7,
    grid=true)

plot!(p_theory_y, [min_val, max_val], [min_val, max_val],
    color=:green, linestyle=:dash, label="Теория: y* = α/β")

savefig(p_theory_y, plotsdir(script_name, "param_scan_theory_y.png"))

# ## 11.4 Тепловая карта: влияние α и β на амплитуду
# 
# Создадим матрицу для тепловой карты

α_vals = sort(unique(results_df.α))
β_vals = sort(unique(results_df.β))
amplitude_matrix = zeros(length(α_vals), length(β_vals))

for i in 1:length(α_vals)
    for j in 1:length(β_vals)
        row = findfirst((results_df.α .== α_vals[i]) .& 
                        (results_df.β .== β_vals[j]))
        if !isnothing(row)
            amplitude_matrix[i, j] = results_df.prey_amplitude[row]
        end
    end
end

p_heatmap = heatmap(β_vals, α_vals, amplitude_matrix,
    xlabel=L"\beta",
    ylabel=L"\alpha",
    title="Тепловая карта: амплитуда жертв",
    color=:viridis,
    colorbar_title="Амплитуда")

savefig(p_heatmap, plotsdir(script_name, "param_scan_heatmap.png"))

# ============================================================
# 12. АНАЛИЗ РЕЗУЛЬТАТОВ ПАРАМЕТРИЧЕСКОГО СКАНИРОВАНИЯ
# ============================================================

println("\n" * "="^60)
println("АНАЛИЗ РЕЗУЛЬТАТОВ ПАРАМЕТРИЧЕСКОГО СКАНИРОВАНИЯ")
println("="^60)

println("\nСтатистика по всем экспериментам:")
println("  Средняя численность жертв: от $(round(minimum(results_df.mean_prey), digits=2)) " *
        "до $(round(maximum(results_df.mean_prey), digits=2))")
println("  Средняя численность хищников: от $(round(minimum(results_df.mean_predator), digits=2)) " *
        "до $(round(maximum(results_df.mean_predator), digits=2))")
println("  Амплитуда жертв: от $(round(minimum(results_df.prey_amplitude), digits=2)) " *
        "до $(round(maximum(results_df.prey_amplitude), digits=2))")
println("  Амплитуда хищников: от $(round(minimum(results_df.predator_amplitude), digits=2)) " *
        "до $(round(maximum(results_df.predator_amplitude), digits=2))")

# Находим эксперименты с максимальной и минимальной амплитудой
max_amp_idx = argmax(results_df.prey_amplitude)
min_amp_idx = argmin(results_df.prey_amplitude)

println("\n🔴 Эксперимент с МАКСИМАЛЬНОЙ амплитудой жертв:")
println("  α = $(results_df.α[max_amp_idx]), β = $(results_df.β[max_amp_idx]), " *
        "δ = $(results_df.δ[max_amp_idx]), γ = $(results_df.γ[max_amp_idx])")
println("  Амплитуда жертв: $(round(results_df.prey_amplitude[max_amp_idx], digits=2))")

println("\n🟢 Эксперимент с МИНИМАЛЬНОЙ амплитудой жертв:")
println("  α = $(results_df.α[min_amp_idx]), β = $(results_df.β[min_amp_idx]), " *
        "δ = $(results_df.δ[min_amp_idx]), γ = $(results_df.γ[min_amp_idx])")
println("  Амплитуда жертв: $(round(results_df.prey_amplitude[min_amp_idx], digits=2))")

# ============================================================
# 13. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

println("\n" * "="^60)
println("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
println("="^60)

# Сохранение графиков базового эксперимента
savefig(plt1, plotsdir(script_name, "lv_dynamics.png"))
savefig(plt2, plotsdir(script_name, "lv_phase_portrait.png"))
savefig(plt3, plotsdir(script_name, "lv_derivatives.png"))
savefig(plt4, plotsdir(script_name, "lv_relative_changes.png"))
savefig(plt5, plotsdir(script_name, "lv_spectrum.png"))
savefig(plt6, plotsdir(script_name, "lv_panel.png"))

println("  📊 Графики базового эксперимента сохранены в: plots/$(script_name)/")

# Сохранение данных базового эксперимента
@save datadir(script_name, "lv_results.jld2") df_lv p_lv x_star y_star
println("  📁 Данные базового эксперимента сохранены в: data/$(script_name)/lv_results.jld2")

# Сохранение результатов параметрического сканирования
CSV.write(datadir(script_name, "parameter_scan_results.csv"), results_df)
println("  📋 Таблица результатов сканирования: data/$(script_name)/parameter_scan_results.csv")

# Сохранение всех данных параметрического сканирования
data_to_save = Dict(
    "all_params" => all_params,
    "all_results" => all_results,
    "results_df" => results_df,
    "param_grid" => param_grid
)
save(datadir(script_name, "lv_parameter_scan_complete.jld2"), data_to_save)
println("  📁 Полные данные сканирования: data/$(script_name)/lv_parameter_scan_complete.jld2")

# ============================================================
# 14. ЗАВЕРШЕНИЕ РАБОТЫ
# ============================================================

println("\n" * "="^60)
println("✅ МОДЕЛИРОВАНИЕ ЗАВЕРШЕНО УСПЕШНО!")
println("="^60)
println("\nВсего выполнено экспериментов: $(length(all_params))")
println("  - Базовый эксперимент: 1")
println("  - Параметрическое сканирование: $(length(all_params))")
println("\nРезультаты сохранены в:")
println("  📊 plots/$(script_name)/ - все графики")
println("  📁 data/$(script_name)/ - все данные")
