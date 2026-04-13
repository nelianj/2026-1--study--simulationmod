# # Базовая визуализация модели Daisyworld
# 
# **Автор:** Ваше Имя
# **Дата:** 2026-03-16
# 
# ## Описание
# 
# Этот скрипт демонстрирует базовую визуализацию модели Daisyworld
# в разные моменты времени: начальное состояние (t=0), после 5 шагов
# и после 40 шагов моделирования.

# ## Подготовка окружения
# 
# Активируем проект DrWatson и загружаем необходимые пакеты:

# ```julia
using DrWatson
@quickactivate "project"

using Agents      # для агентного моделирования
using DataFrames  # для работы с данными
using Plots       # для визуализации
# ```

# ## Подключение модели
# 
# Подключаем файл с реализацией модели Daisyworld

# ```julia
include(srcdir("daisyworld.jl"))
# ```

# ## Создание модели
# 
# Инициализируем мир Daisyworld с параметрами по умолчанию

# ```julia
model = daisyworld()
# ```

# ## Функция для визуализации

# ```julia
function plot_daisyworld(model, title_str)
    p1 = heatmap(model.temperature', 
                title = "Температура поверхности\n$title_str",
                xlabel = "X", ylabel = "Y",
                color = :thermal, clims = (-20, 60))
    
    black_x, black_y, white_x, white_y = Int[], Int[], Int[], Int[]
    for agent in allagents(model)
        if agent.breed == :black
            push!(black_x, agent.pos[1]); push!(black_y, agent.pos[2])
        else
            push!(white_x, agent.pos[1]); push!(white_y, agent.pos[2])
        end
    end
    
    !isempty(black_x) && scatter!(p1, black_x, black_y, color=:black, marker=:star5, label="Чёрные")
    !isempty(white_x) && scatter!(p1, white_x, white_y, color=:white, marker=:star5, label="Белые")
    
    return p1
end
# ```

# ## Визуализация начального состояния (t = 0)

# ```julia
plt1 = plot_daisyworld(model, "t = 0")
savefig(plt1, plotsdir("daisy_step001.png"))
# ```

# ## Моделирование 5 шагов

# ```julia
step!(model, 5)
plt2 = plot_daisyworld(model, "t = 5")
savefig(plt2, plotsdir("daisy_step005.png"))
# ```

# ## Моделирование до 40 шагов

# ```julia
step!(model, 35)
plt3 = plot_daisyworld(model, "t = 40")
savefig(plt3, plotsdir("daisy_step040.png"))
# ```

# ## Результаты
# 
# Графики сохранены в каталоге `plots/`:
# - `daisy_step001.png` - начальное состояние
# - `daisy_step005.png` - после 5 шагов
# - `daisy_step040.png` - после 40 шагов
