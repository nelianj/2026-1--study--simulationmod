using DrWatson
@quickactivate "project"
include(srcdir("DiningPhilosophers.jl"))
using .DiningPhilosophers
using DataFrames, CSV, Plots

N = 2
N1 = 10
tmax = 50.0

println("=== Классическая сеть (без арбитра) ===")
net_classic, u0_classic, _ = build_classical_network(N)
net_classic1, u0_classic1, _ = build_classical_network(N1)
df_classic = simulate_stochastic(net_classic, u0_classic, tmax)
df_classic1 = simulate_stochastic(net_classic1, u0_classic1, tmax)

CSV.write(datadir("dining_classic(N=2).csv"), df_classic)
CSV.write(datadir("dining_classic(N=10).csv"), df_classic1)

dead = detect_deadlock(df_classic, net_classic)
dead1 = detect_deadlock(df_classic1, net_classic1)
println("Deadlock обнаружен(N=2): $dead")
println("Deadlock обнаружен(N=10): $dead1")

plot_classic = plot_marking_evolution(df_classic, N)
plot_classic1 = plot_marking_evolution(df_classic1, N1)
savefig(plotsdir("classic_simulation(N=2).png"))
savefig(plotsdir("classic_simulation(N=10).png"))

println("\n=== Сеть с арбитром ===")
net_arb, u0_arb, _ = build_arbiter_network(N)
df_arb = simulate_stochastic(net_arb, u0_arb, tmax)
CSV.write(datadir("dining_arbiter(N=2).csv"), df_arb)
dead_arb = detect_deadlock(df_arb, net_arb)
println("Deadlock обнаружен(N=2): $dead_arb")
plot_arb = plot_marking_evolution(df_arb, N)
savefig(plotsdir("arbiter_simulation(N=2).png"))

net_arb1, u0_arb1, _ = build_arbiter_network(N1)
df_arb1 = simulate_stochastic(net_arb1, u0_arb1, tmax)
CSV.write(datadir("dining_arbiter(N=10).csv"), df_arb1)
dead_arb1 = detect_deadlock(df_arb1, net_arb1)
println("Deadlock обнаружен(N=10): $dead_arb1")
plot_arb1 = plot_marking_evolution(df_arb1, N1)
savefig(plotsdir("arbiter_simulation(N=10).png"))
