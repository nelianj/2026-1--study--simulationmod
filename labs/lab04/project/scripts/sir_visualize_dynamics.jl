# # Комплексный анализ результатов сканирования β
#
# Загружаем результаты параметрического сканирования и строим
# три графика, показывающих зависимость различных показателей
# эпидемии от коэффициента заразности β.

using DrWatson
@quickactivate "project"
using Agents, DataFrames, Plots, CSV

include(srcdir("sir_model.jl"))

# ## Загрузка данных
#
# Читаем CSV-файл с результатами всех экспериментов,
# где для каждого β и seed были рассчитаны показатели.

df = CSV.read(datadir("beta_scan_all.csv"), DataFrame)

# ## Визуализация результатов
#
# Создаём три графика:
# - **Пик эпидемии** и **конечная доля инфицированных** от β
# - **Число умерших** от β
# - **Доля выздоровевших** от β

p1 = plot(df.beta, df.peak, label = "Пик", xlabel = "β", ylabel = "Доля инфицированных")
plot!(p1, df.beta, df.final_inf, label = "Конечная")

p2 = plot(df.beta, df.deaths, xlabel = "β", ylabel = "Число умерших")

p3 = plot(df.beta, df.final_rec, xlabel = "β", ylabel = "Доля выздоровевших")

# ## Сохранение графика
#
# Объединяем три графика в один вертикальный макет
# и сохраняем в файл.

plot(p1, p2, p3, layout = (3, 1), size = (800, 900))
savefig(plotsdir("comprehensive_analysis.png"))
