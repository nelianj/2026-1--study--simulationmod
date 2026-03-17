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

temperature(model) = StatsBase.mean(model.temperature)

mdata = [temperature, :solar_luminosity]

param_dict = Dict(
    :griddims => (30, 30),           # размер сетки 30×30
    :init_black => 0.2,               # начальная доля чёрных маргариток (20%)
    :albedo_white => 0.75,             # альбедо белых маргариток
    :albedo_black => 0.25,             # альбедо чёрных маргариток
    :surface_albedo => 0.4,            # альбедо пустой поверхности
    :solar_change => 0.005,            # скорость изменения светимости
    :solar_luminosity => 1.0,          # начальная светимость
    :scenario => :ramp,                 # сценарий с изменением светимости
    :seed => 165,                       # зерно для генератора случайных чисел
    :max_age => [25, 40],               # максимальный возраст (два значения)
    :init_white => [0.2, 0.8],           # начальная доля белых (20% и 80%)
)

println("Параметры эксперимента:")
println("   - Сценарий: :ramp (изменение солнечной активности)")
println("   - Варьируемые параметры: max_age ∈ [25, 40], init_white ∈ [0.2, 0.8]")
println("   - Фиксированные параметры: сетка 30×30, init_black = 0.2")
println("   - Всего комбинаций: $(length(param_dict[:max_age]) * length(param_dict[:init_white]))")

params_list = dict_list(param_dict)
println("Сгенерировано комбинаций параметров: ", length(params_list))

println("ЗАПУСК ПАРАМЕТРИЧЕСКОГО ИССЛЕДОВАНИЯ (СЦЕНАРИЙ :ramp)")

for (i, params) in enumerate(params_list)
    println("\nЭксперимент $i/$(length(params_list))")
    println("   Параметры: max_age = $(params[:max_age]), init_white = $(params[:init_white])")

    model = daisyworld(; params...)# Создание модели с текущими параметрами
    println("Модель создана")
    println("Запуск моделирования на 1000 шагов (сценарий :ramp)...")
    agent_df, model_df = run!(model, 1000; adata = adata, mdata = mdata)# Запуск моделирования. Выполняем 1000 шагов и собираем данные о популяциях, температуре и солнечной светимости.
    println("Моделирование завершено")
    println("      - Собрано $(nrow(agent_df)) записей о популяциях")
    println("      - Собрано $(nrow(model_df)) записей о состоянии модели")

    figure = CairoMakie.Figure(size = (600, 600))# Создание комплексной визуализации. 1. Динамика численности маргариток. 2. Изменение средней температуры. 3. Изменение солнечной светимост

    ax1 = Axis(figure[1, 1],
               ylabel = "Количество маргариток",
               title = "Динамика популяций (max_age=$(params[:max_age]), init_white=$(params[:init_white]))")# График 1: Динамика численности маргариток
    black_line = lines!(ax1, agent_df[!, :time], agent_df[!, :count_black],
                        color = :red, linewidth = 2, label = "Чёрные")
    white_line = lines!(ax1, agent_df[!, :time], agent_df[!, :count_white],
                        color = :blue, linewidth = 2, label = "Белые")# Линии для чёрных и белых маргариток

    figure[1, 2] = Legend(figure, [black_line, white_line],
                          ["Чёрные", "Белые"], labelsize = 12)# Добавляем легенду справа от первого графика

    ax2 = Axis(figure[2, 1],
               ylabel = "Средняя температура (°C)")# График 2: Динамика средней температуры

    lines!(ax2, model_df[!, :time], model_df[!, :temperature],
           color = :red, linewidth = 2)

    ax3 = Axis(figure[3, 1],
               xlabel = "Время (шаги)",
               ylabel = "Солнечная светимость")# График 3: Динамика солнечной светимости

    lines!(ax3, model_df[!, :time], model_df[!, :solar_luminosity],
           color = :red, linewidth = 2)

    for ax in (ax1, ax2)
        ax.xticklabelsvisible = false
        ax.xlabel = ""
    end # Скрываем подписи оси X на верхних графиках для компактности
    display(figure)
    plt_name = savename("daisy-luminosity", params) * ".png"
    save(plotsdir(plt_name), figure)# Сохранение графика

    println("График сохранён: $(plt_name)")
end

println("ИТОГИ ПАРАМЕТРИЧЕСКОГО ИССЛЕДОВАНИЯ (СЦЕНАРИЙ :ramp)")
println("Всего выполнено экспериментов: ", length(params_list))
println("Всего создано графиков: ", length(params_list))
println("Графики сохранены в каталоге: ", plotsdir())
