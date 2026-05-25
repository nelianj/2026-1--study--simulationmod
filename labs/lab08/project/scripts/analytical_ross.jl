using DrWatson
@quickactivate "project"

using Plots, DataFrames, CSV

function analytical_crash_time(N::Int, S::Int, c::Int, λ::Float64, μ::Float64)
    M = N + S  # total functional machines
    T = zeros(Float64, M + 1)  # T[k] = time to crash from state k
    T[1] = 0.0  # k=0 functional machines -> crash
    
    for k in 1:M
        # Death rate (failures)
        death_rate = λ * min(k, N)
        # Birth rate (repairs)
        birth_rate = μ * min(c, M - k)
        total_rate = death_rate + birth_rate
        
        if k == 1
            # Only death possible (no machines to repair if all failed)
            T[k+1] = 1/death_rate + T[k]
        elseif k == M
            # Only birth possible (all machines functional, no failures possible from spares)
            T[k+1] = T[k]
        else
            # General case
            death_prob = death_rate / total_rate
            birth_prob = birth_rate / total_rate
            
            # Solve: T[k] = 1/total_rate + death_prob * T[k-1] + birth_prob * T[k+1]
            # Rearranged for T[k+1]:
            T[k+1] = (T[k] - 1/total_rate - death_prob * T[k-1]) / birth_prob
        end
    end
    
    return T[M+1]  # starting from k = M functional machines
end

# Parameters
λ = 0.01  # 1/100 hours
μ = 1.0   # 1 repair per hour

# Configurations to test
configs = [(10, 3, 1), (15, 5, 1), (20, 7, 1), (10, 3, 2), (10, 3, 3)]
results_analytical = []

println("ANALYTICAL SOLUTION COMPARISON")

for (N, S, c) in configs
    T_crash = analytical_crash_time(N, S, c, λ, μ)
    push!(results_analytical, (N=N, S=S, c=c, analytical=T_crash))
    println("N=$N, S=$S, c=$c → Analytical crash time: $(round(T_crash, digits=2)) hours")
end

df_analytical = DataFrame(results_analytical)

# Your simulation results (from previous runs)
simulation_results = [
    (N=10, S=3, c=1, sim=5000.0),
    (N=15, S=5, c=1, sim=100000.0),  # approximate from your graph
    (N=20, S=7, c=1, sim=60000.0),
    (N=10, S=3, c=2, sim=65000.0),
    (N=10, S=3, c=3, sim=140000.0),
]

df_simulation = DataFrame(simulation_results)

# Merge for comparison
df_compare = innerjoin(df_analytical, df_simulation, on=[:N, :S, :c])
df_compare[!, :ratio] = df_compare.sim ./ df_compare.analytical

println("SIMULATION vs ANALYTICAL")
for row in eachrow(df_compare)
    println("N=$(row.N), S=$(row.S), c=$(row.c):")
    println("  Analytical: $(round(row.analytical, digits=2)) hours")
    println("  Simulation: $(round(row.sim, digits=2)) hours")
    println("  Ratio (Sim/Anal): $(round(row.ratio, digits=3))")
    println()
end

# Plot comparison
p1 = scatter(df_compare.N, df_compare.analytical, 
    label="Analytical", marker=:circle, markersize=8, color=:blue)
scatter!(p1, df_compare.N, df_compare.sim, 
    label="Simulation", marker=:square, markersize=8, color=:red)
plot!(p1, xlabel="Number of working machines (N)", 
    ylabel="Mean time to crash (hours)",
    title="Ross Model: Simulation vs Analytical Solution",
    legend=:topright)

savefig(plotsdir("ross_analytical_comparison.png"))

# Create comparison table
df_comparison_table = DataFrame(
    N = df_compare.N,
    S = df_compare.S,
    Repairmen = df_compare.c,
    Analytical = round.(df_compare.analytical, digits=0),
    Simulation = round.(df_compare.sim, digits=0),
    Ratio = round.(df_compare.ratio, digits=2)
)

CSV.write(datadir("ross_analytical_comparison.csv"), df_comparison_table)

println("\nResults saved to:")
println("  - plots/ross_analytical_comparison.png")
println("  - data/ross_analytical_comparison.csv")
