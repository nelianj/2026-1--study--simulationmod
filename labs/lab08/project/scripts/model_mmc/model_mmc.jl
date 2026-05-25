using DrWatson
@quickactivate "project"

using StableRNGs
using Distributions
using ConcurrentSim
using ResumableFunctions
using DataFrames
using Plots
using CSV
gr(fmt=:png)

rng = StableRNG(123)
num_customers = 10
num_servers = 2
mu = 1.0 / 2
lam = 0.9
arrival_dist = Exponential(1 / lam)
service_dist = Exponential(1 / mu)

history = Vector{Tuple{Float64, String, Int}}()

@resumable function customer(
    env::Environment,
    server::Resource,
    id::Integer,
    t_a::Float64,
    d_s::Distribution,
)
    @yield timeout(env, t_a)
    println("Customer $id arrived: ", now(env))
    push!(history, (now(env), "arrived", id))

    @yield request(server)
    println("Customer $id entered service: ", now(env))
    push!(history, (now(env), "started", id))

    @yield timeout(env, rand(rng, d_s))
    @yield unlock(server)
    println("Customer $id exited service: ", now(env))
    push!(history, (now(env), "exited", id))
end

function setup_and_run()
    sim = Simulation()
    server = Resource(sim, num_servers)
    arrival_time = 0.0

    for i = 1:num_customers
        arrival_time += rand(rng, arrival_dist)
        @process customer(sim, server, i, arrival_time, service_dist)
    end

    run(sim)
end

setup_and_run()

times = sort(unique([t for (t, _, _) in history]))
system_count = zeros(Int, length(times))
queue_count = zeros(Int, length(times))
server_usage = zeros(Float64, length(times))

for (idx, t) in enumerate(times)
    arrived = count(x -> x[1] <= t && x[2] == "arrived", history)
    exited = count(x -> x[1] <= t && x[2] == "exited", history)
    started = count(x -> x[1] <= t && x[2] == "started", history)

    system_count[idx] = arrived - exited
    queue_count[idx] = arrived - started
    server_usage[idx] = min(1.0, started / num_servers)
end

df_results = DataFrame(
    time = times,
    n_in_system = system_count,
    queue_length = queue_count,
    utilization = server_usage
)

mkpath(datadir())
mkpath(plotsdir())
CSV.write(datadir("mmc_results.csv"), df_results)

p1 = plot(df_results.time, df_results.n_in_system,
    label = "Клиентов в системе",
    xlabel = "Время",
    ylabel = "Количество клиентов",
    title = "M/M/c: Загрузка системы во времени",
    linewidth = 2,
    legend = :topright)
display(p1)
savefig(plotsdir("mmc_system_load.png"))

p2 = plot(df_results.time, df_results.queue_length,
    label = "Длина очереди",
    xlabel = "Время",
    ylabel = "Клиентов ожидает",
    title = "M/M/c: Динамика очереди",
    linewidth = 2,
    legend = :topright)
display(p2)
savefig(plotsdir("mmc_queue_length.png"))

ρ = lam / (num_servers * mu)
p3 = plot(df_results.time, df_results.utilization,
    label = "Использование",
    xlabel = "Время",
    ylabel = "Использование (0-1)",
    title = "M/M/c: Загрузка серверов",
    linewidth = 2,
    legend = :topright,
    ylims = (0, 1))
hline!([ρ], label = "Теоретический ρ", linestyle = :dash)
display(p3)
savefig(plotsdir("mmc_utilization.png"))
