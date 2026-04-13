using DrWatson
@quickactivate "project"
using Agents, DataFrames, Plots
using Plots
gr(fmt=:png)
using JLD2

include(srcdir("sir_model.jl"))

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

model = initialize_sir(; params...)

times = Int[]
S_vals = Int[]
I_vals = Int[]
R_vals = Int[]
total_vals = Int[]

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

println("Общая численность населения: $(total_vals[end])")
println("Переболело: $(R_vals[end]) человек ($(round(R_vals[end]/1000*100, digits=1))%)")
println("Умерло: $(total_vals[1] - total_vals[end]) человек")
println("Максимальное число инфицированных: $(maximum(I_vals))")

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

savefig(plotsdir("sir_basic_dynamics.png"))
println("\nГрафик сохранён")

@save datadir("sir_basic_agent.jld2") agent_df
@save datadir("sir_basic_model.jld2") model_df

println("Данные сохранены")
