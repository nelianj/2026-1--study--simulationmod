using DrWatson
@quickactivate "project"

using Agents      # для агентного моделирования
using DataFrames  # для работы с данными
using Plots       # для визуализации
using CairoMakie  # для высококачественной графики

include(srcdir("daisyworld.jl"))

black(a::Daisy) = a.breed == :black
white(a::Daisy) = a.breed == :white

adata = [(black, count), (white, count)]

param_dict = Dict(# Фиксированные параметры
    :griddims => (30, 30),           # размер сетки 30×30
    :init_black => 0.2,               # начальная доля чёрных маргариток (20%)
    :albedo_white => 0.75,             # альбедо белых маргариток
    :albedo_black => 0.25,             # альбедо чёрных маргариток
    :surface_albedo => 0.4,            # альбедо пустой поверхности
    :solar_change => 0.005,            # скорость изменения светимости
    :solar_luminosity => 1.0,          # начальная светимость
    :scenario => :default,              # сценарий (без изменения светимости)
    :seed => 165,                       # зерно для генератора случайных чисел
    :max_age => [25, 40],               # максимальный возраст (два значения)
    :init_white => [0.2, 0.8],           # начальная доля белых (20% и 80%)
)

println("Параметры эксперимента:")
println("   - Варьируемые параметры: max_age ∈ [25, 40], init_white ∈ [0.2, 0.8]")
println("   - Фиксированные параметры: сетка 30×30, init_black = 0.2")
println("   - Всего комбинаций: $(length(param_dict[:max_age]) * length(param_dict[:init_white]))")

params_list = dict_list(param_dict)
println("Сгенерировано комбинаций параметров: ", length(params_list))

println("ЗАПУСК ПАРАМЕТРИЧЕСКОГО ИССЛЕДОВАНИЯ")

for (i, params) in enumerate(params_list)
    println("\nЭксперимент $i/$(length(params_list))")
    println("   Параметры: max_age = $(params[:max_age]), init_white = $(params[:init_white])")

    model = daisyworld(; params...)# Создание модели с текущими параметрами
    println("Модель создана")

    println("Запуск моделирования на 1000 шагов...")# Запуск моделирования. Выполняем 1000 шагов и собираем данные о популяциях.
    agent_df, model_df = run!(model, 1000; adata)
    println("Моделирование завершено")
    println("      - Собрано $(nrow(agent_df)) записей")
    println("      - Финальная численность: чёрные = $(agent_df.count_black[end]), белые = $(agent_df.count_white[end])")

    figure = Figure(size = (600, 400))# Создание визуализации

    ax = Axis(figure[1, 1], # Создаём оси с подписями
              xlabel = "Время (шаги)",
              ylabel = "Количество маргариток",
              title = "Динамика численности (max_age=$(params[:max_age]), init_white=$(params[:init_white]))")# Создаём оси с подписями

    black_line = lines!(ax, agent_df[!, :time],
                        color = :black, linewidth = 2, label = "Чёрные")
    white_line = lines!(ax, agent_df[!, :time], agent_df[!, :count_white],
                        color = :orange, linewidth = 2, label = "Белые")# Строим линии для чёрных и белых маргариток agent_df[!, :count_black]

    Legend(figure[1, 2], [black_line, white_line], ["Чёрные", "Белые"], labelsize = 12) # Добавляем легенду

    display(figure)

    plt_name = savename("daisy-count", params) * ".png"
    save(plotsdir(plt_name), figure)# Сохранение графика

    println("График сохранён: $(plt_name)")
end
