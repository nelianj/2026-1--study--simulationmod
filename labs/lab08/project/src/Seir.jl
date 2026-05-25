using ResumableFunctions, ConcurrentSim, Distributions, DataFrames, Random

# Вспомогательные функции для обновления массивов состояния
function increment!(a::Array{Int64})
    push!(a, a[length(a)] + 1)
end
function decrement!(a::Array{Int64})
    push!(a, a[length(a)] - 1)
end
function carryover!(a::Array{Int64})
    push!(a, a[length(a)])
end

# Структуры данных для SEIR
mutable struct SEIRPerson
    id::Int64
    status::Symbol # :S, :E, :I, :R
end

mutable struct SEIRModel
    sim::ConcurrentSim.Simulation
    β::Float64
    c::Float64
    γ::Float64
    σ::Float64
    ta::Array{Float64}
    Sa::Array{Int64}
    Ea::Array{Int64}
    Ia::Array{Int64}
    Ra::Array{Int64}
    allIndividuals::Array{SEIRPerson}
end

# Функции обновления статистики при событиях для SEIR
function seir_infection_update!(sim::ConcurrentSim.Simulation, m::SEIRModel)
    push!(m.ta, ConcurrentSim.now(sim))
    decrement!(m.Sa)
    increment!(m.Ea)
    carryover!(m.Ia)
    carryover!(m.Ra)
end

function seir_activation_update!(sim::ConcurrentSim.Simulation, m::SEIRModel)
    push!(m.ta, ConcurrentSim.now(sim))
    carryover!(m.Sa)
    decrement!(m.Ea)
    increment!(m.Ia)
    carryover!(m.Ra)
end

function seir_recovery_update!(sim::ConcurrentSim.Simulation, m::SEIRModel)
    push!(m.ta, ConcurrentSim.now(sim))
    carryover!(m.Sa)
    carryover!(m.Ea)
    decrement!(m.Ia)
    increment!(m.Ra)
end

# Основная логика жизни индивида для SEIR
@resumable function live_seir(env::ConcurrentSim.Simulation, individual::SEIRPerson, m::SEIRModel)
    while individual.status == :S
        @yield timeout(env, rand(Exponential(1/m.c)))
        alter = individual
        while alter == individual
            N = length(m.allIndividuals)
            index = rand(DiscreteUniform(1, N))
            alter = m.allIndividuals[index]
        end
        if alter.status == :I
            if rand(Uniform(0, 1)) < m.β
                individual.status = :E
                seir_infection_update!(env, m)
            end
        end
    end
    if individual.status == :E
        @yield timeout(env, rand(Exponential(1/m.σ)))
        individual.status = :I
        seir_activation_update!(env, m)
    end
    if individual.status == :I
        @yield timeout(env, rand(Exponential(1/m.γ)))
        individual.status = :R
        seir_recovery_update!(env, m)
    end
end

# Функции создания и запуска модели SEIR
function MakeSEIRModel(u0, p, σ)
    (S, E, I, R) = u0
    N = S + E + I + R
    (β, c, γ) = p
    sim = ConcurrentSim.Simulation()
    allIndividuals = SEIRPerson[]
    for i = 1:S
        push!(allIndividuals, SEIRPerson(i, :S))
    end
    for i = (S+1):(S+E)
        push!(allIndividuals, SEIRPerson(i, :E))
    end
    for i = (S+E+1):(S+E+I)
        push!(allIndividuals, SEIRPerson(i, :I))
    end
    for i = (S+E+I+1):N
        push!(allIndividuals, SEIRPerson(i, :R))
    end
    ta = Float64[0.0]
    Sa = Int64[S]
    Ea = Int64[E]
    Ia = Int64[I]
    Ra = Int64[R]
    SEIRModel(sim, β, c, γ, σ, ta, Sa, Ea, Ia, Ra, allIndividuals)
end

function activate_seir(m::SEIRModel)
    [@process live_seir(m.sim, individual, m) for individual in m.allIndividuals]
end

function seir_run(m::SEIRModel, tf::Float64)
    ConcurrentSim.run(m.sim, tf)
end

function out_seir(m::SEIRModel)
    result = DataFrame()
    result[!, :t] = m.ta
    result[!, :S] = m.Sa
    result[!, :E] = m.Ea
    result[!, :I] = m.Ia
    result[!, :R] = m.Ra
    return result
end
