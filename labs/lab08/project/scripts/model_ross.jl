# # Модель Росса (Система с резервированием и ремонтом)
# 
# В системе находятся N работающих машин и S резервных. При поломке машина 
# заменяется резервной (если есть) и отправляется в ремонт. Если резерва нет — 
# система падает. Ремонт выполняют несколько ремонтников.

# ## Загрузка необходимых пакетов

using DrWatson
@quickactivate "project"

using ResumableFunctions
using ConcurrentSim
using Distributions
using Random
using StableRNGs
using DataFrames
using CSV
using Plots
using Statistics
gr(fmt=:png)

# ## Параметры симуляции

const RUNS = 5
const N = 10
const S = 3
const LAMBDA = 100.0
const MU = 1.0
const SEED = 42

rng = StableRNG(SEED)
F = Exponential(LAMBDA)
G = Exponential(1/MU)

# ## Структура для хранения состояния системы

mutable struct SystemState
    operational::Int
    in_repair::Int
    spares::Int
    history::Vector{Tuple{Float64, Int, Int, Int}}
end

# ## Поведение машины

@resumable function machine(
    env::Environment,
    repair_queue::Resource,
    state::SystemState,
    machine_id::Int,
)
    while true
        @yield timeout(env, rand(rng, F))
        
        if state.operational > 0
            state.operational -= 1
        end
        
        if state.spares > 0
            state.spares -= 1
            state.operational += 1
        else
            throw(StopSimulation("No more spares!"))
        end
        
        push!(state.history, (now(env), state.operational, state.in_repair, state.spares))
        
        state.in_repair += 1
        push!(state.history, (now(env), state.operational, state.in_repair, state.spares))
        
        @yield request(repair_queue)
        @yield timeout(env, rand(rng, G))
        @yield unlock(repair_queue)
        
        state.in_repair -= 1
        state.spares += 1
        push!(state.history, (now(env), state.operational, state.in_repair, state.spares))
    end
end

# ## Запуск симуляции

function sim_repair(N::Int, S::Int, num_repairmen::Int)
    sim = Simulation()
    repair_queue = Resource(sim, num_repairmen)
    
    state = SystemState(N, 0, S, Vector{Tuple{Float64, Int, Int, Int}}())
    push!(state.history, (0.0, N, 0, S))
    
    for i in 1:N
        @process machine(sim, repair_queue, state, i)
    end
    
    for i in 1:S
        @process machine(sim, repair_queue, state, N + i)
    end
    
    try
        run(sim)
        return now(sim), state.history
    catch e
        if isa(e, StopSimulation)
            msg = e.msg
            stop_time = now(sim)
            println("At time $stop_time: $msg")
            return stop_time, state.history
        else
            rethrow()
        end
    end
end

# ## Построение DataFrame из истории

function history_to_dataframe(history::Vector{Tuple{Float64, Int, Int, Int}})
    times = [t for (t, _, _, _) in history]
    operational = [op for (_, op, _, _) in history]
    in_repair = [rep for (_, _, rep, _) in history]
    spares = [sp for (_, _, _, sp) in history]
    
    all_times = sort(unique(vcat(times, 0.0:1.0:maximum(times))))
    op_continuous = zeros(Int, length(all_times))
    rep_continuous = zeros(Int, length(all_times))
    spare_continuous = zeros(Int, length(all_times))
    
    for (idx, t) in enumerate(all_times)
        last_idx = findlast(ti -> ti <= t, times)
        if last_idx !== nothing
            op_continuous[idx] = operational[last_idx]
            rep_continuous[idx] = in_repair[last_idx]
            spare_continuous[idx] = spares[last_idx]
        elseif idx > 1
            op_continuous[idx] = op_continuous[idx-1]
            rep_continuous[idx] = rep_continuous[idx-1]
            spare_continuous[idx] = spare_continuous[idx-1]
        end
    end
    
    return DataFrame(
        time = all_times,
        operational = op_continuous,
        in_repair = rep_continuous,
        spares = spare_continuous
    )
end

# ## Создание каталогов

mkpath(datadir())
mkpath(plotsdir())

# ## Эксперимент 1: Базовый прогон (как в оригинале)

results_base = Float64[]

for i in 1:RUNS
    crash_time, _ = sim_repair(N, S, 1)
    push!(results_base, crash_time)
end

avg_crash_base = sum(results_base) / RUNS
println("\nAverage crash time: $avg_crash_base")

# ## График 1: Динамика исправных машин (базовый прогон)

_, history_base = sim_repair(N, S, 1)
df_dynamics_base = history_to_dataframe(history_base)

p1 = plot(df_dynamics_base.time, df_dynamics_base.operational,
    marker = :circle, markersize = 3, linewidth = 2,
    label = "Работающие машины",
    xlabel = "Время", ylabel = "Количество машин",
    title = "Модель Росса: N=$N, S=$S, ремонтников=1")
plot!(p1, df_dynamics_base.time, df_dynamics_base.spares,
    label = "Резервные машины", linewidth = 2, linestyle = :dash)
plot!(p1, df_dynamics_base.time, df_dynamics_base.in_repair,
    label = "Машины в ремонте", linewidth = 2, linestyle = :dot)
hline!([N], label = "Требуется работающих (N)", linestyle = :dashdot)
savefig(plotsdir("ross_dynamics_base.png"))

# ## Эксперимент 2: Разное количество ремонтников
println("ЭКСПЕРИМЕНТ 1: Влияние количества ремонтников")

num_repairmen_list = [1, 2, 3]
results_repairmen = []

for num_rep in num_repairmen_list
    println("\nРемонтников: $num_rep")
    crash_times = Float64[]
    for run in 1:RUNS
        crash_time, _ = sim_repair(N, S, num_rep)
        push!(crash_times, crash_time)
    end
    avg_time = mean(crash_times)
    push!(results_repairmen, (num_repairmen = num_rep, avg_crash_time = avg_time))
    println("  Среднее время краха: $(round(avg_time, digits=2))")
end

df_repairmen = DataFrame(results_repairmen)
p2 = plot(df_repairmen.num_repairmen, df_repairmen.avg_crash_time,
    marker = :circle, markersize = 8, linewidth = 2,
    label = "Среднее время краха",
    xlabel = "Количество ремонтников",
    ylabel = "Время до краха",
    title = "Модель Росса: Влияние числа ремонтников")
savefig(plotsdir("ross_repairmen_impact.png"))

# ## Эксперимент 3: Разные конфигурации машин

println("\n")
println("ЭКСПЕРИМЕНТ 2: Разные конфигурации (N, S)")

machine_configs = [(10, 3), (15, 5), (20, 7)]
results_configs = []

for (N_val, S_val) in machine_configs
    println("\nКонфигурация: N=$N_val, S=$S_val")
    crash_times = Float64[]
    for run in 1:RUNS
        crash_time, _ = sim_repair(N_val, S_val, 1)
        push!(crash_times, crash_time)
    end
    avg_time = mean(crash_times)
    push!(results_configs, (N = N_val, S = S_val, avg_crash_time = avg_time))
    println("  Среднее время краха: $(round(avg_time, digits=2))")
end

df_configs = DataFrame(results_configs)
p3 = plot(df_configs.N, df_configs.avg_crash_time,
    marker = :square, markersize = 8, linewidth = 2,
    label = "Среднее время краха",
    xlabel = "Число работающих машин (N)",
    ylabel = "Время до краха",
    title = "Модель Росса: Влияние N при фиксированном S")
savefig(plotsdir("ross_config_comparison.png"))

# ## Сохранение всех результатов

df_summary = vcat(
    DataFrame(experiment = "Ремонтники", parameter = df_repairmen.num_repairmen, avg_time = df_repairmen.avg_crash_time),
    DataFrame(experiment = "Конфигурации", parameter = df_configs.N, avg_time = df_configs.avg_crash_time)
)
CSV.write(datadir("ross_results_summary.csv"), df_summary)
CSV.write(datadir("ross_dynamics_base.csv"), df_dynamics_base)
println("РЕЗУЛЬТАТЫ СОХРАНЕНЫ")
