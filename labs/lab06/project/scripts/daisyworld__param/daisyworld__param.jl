using DrWatson
@quickactivate "project"

using Agents      # для агентного моделирования
using DataFrames  # для работы с данными
using Plots       # для визуализации
using CairoMakie  # для высококачественной графики

include(srcdir("daisyworld.jl"))

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
println("   - Варьируемые параметры: max_age = [25, 40], init_white = [0.2, 0.8]")
println("   - Фиксированные параметры: сетка 30×30, альбедо чёрных = 0.25, альбедо белых = 0.75")

params_list = dict_list(param_dict)
println("Всего комбинаций параметров: ", length(params_list))

daisycolor(a::Daisy) = a.breed  # :black или :white

plotkwargs = (
    agent_color = daisycolor,          # цвет агентов (чёрный/белый)
    agent_size = 20,                    # размер маркеров
    agent_marker = '✿',                  # символ для маргариток (цветок)
    heatarray = :temperature,            # данные для тепловой карты
    heatkwargs = (colorrange = (-20, 60),),  # диапазон температур
)

println("ЗАПУСК ПАРАМЕТРИЧЕСКОГО ИССЛЕДОВАНИЯ")

for (i, params) in enumerate(params_list)
    println("\nЭксперимент $i/$(length(params_list))")
    println("   Параметры: max_age = $(params[:max_age]), init_white = $(params[:init_white])")

    model = daisyworld(; params...)# Создание модели с текущими параметрами
    println("Модель создана")

    println("Визуализация t = 0...") # Визуализация начального состояния (t = 0)
    plt1, _ = abmplot(model; plotkwargs...)
    fig1 = Figure(size = (800, 600))
    ax1 = Axis(fig1[1,1], title = "Начальное состояние (t=0)\nmax_age=$(params[:max_age]), init_white=$(params[:init_white])")
    display(plt1)

    step!(model, 5)# Моделирование 5 шагов
    println("Визуализация t = 5...")
    plt2, _ = abmplot(model; heatarray = model.temperature, plotkwargs...)
    display(plt2)
    step!(model, 35)  # Моделирование до 40 шагов
    println("Визуализация t = 40...")
    plt3, _ = abmplot(model; heatarray = model.temperature, plotkwargs...)
    display(plt3)

    plt1_name = savename("daisyworld", params) * "_step01.png" # Формирование имён файлов
    plt2_name = savename("daisyworld", params) * "_step05.png"
    plt3_name = savename("daisyworld", params) * "_step40.png"

    save(plotsdir(plt1_name), plt1)#Сохранение графиков
    save(plotsdir(plt2_name), plt2)
    save(plotsdir(plt3_name), plt3)

    println("Графики сохранены:")
    println("      - $(plt1_name)")
    println("      - $(plt2_name)")
    println("      - $(plt3_name)")
end

println("ИТОГИ ПАРАМЕТРИЧЕСКОГО ИССЛЕДОВАНИЯ")
println("Всего выполнено экспериментов: ", length(params_list))
println("Всего создано графиков: ", length(params_list) * 3)
println("Графики сохранены в каталоге: ", plotsdir())
