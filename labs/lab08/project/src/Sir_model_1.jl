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

# Структуры данных
mutable struct SIRPerson
    id::Int64
    status::Symbol # :S, :I, :R
end

mutable struct SIRModel
    sim::ConcurrentSim.Simulation # Тип Simulation, не Environment
    β::Float64
    c::Float64
    γ::Float64
    μ::Float64
    ta::Array{Float64}
    Sa::Array{Int64}
    Ia::Array{Int64}
    Ra::Array{Int64}
    allIndividuals::Array{SIRPerson}
end

# Функции обновления статистики при событиях
function infection_update!(sim::ConcurrentSim.Simulation, m::SIRModel)
    push!(m.ta, ConcurrentSim.now(sim))
    decrement!(m.Sa)
    increment!(m.Ia)
    carryover!(m.Ra)
end

function recovery_update!(sim::ConcurrentSim.Simulation, m::SIRModel)
    push!(m.ta, ConcurrentSim.now(sim))
    carryover!(m.Sa)
    decrement!(m.Ia)
    increment!(m.Ra)
end

function death_update!(sim::ConcurrentSim.Simulation, m::SIRModel, status::Symbol)
    push!(m.ta, ConcurrentSim.now(sim))
    if status == :S
        decrement!(m.Sa)
        carryover!(m.Ia)
        carryover!(m.Ra)
    elseif status == :I
        carryover!(m.Sa)
        decrement!(m.Ia)
        carryover!(m.Ra)
    elseif status == :R
        carryover!(m.Sa)
        carryover!(m.Ia)
        decrement!(m.Ra)
    end
end

function birth_update!(sim::ConcurrentSim.Simulation, m::SIRModel)
    push!(m.ta, ConcurrentSim.now(sim))
    increment!(m.Sa)
    carryover!(m.Ia)
    carryover!(m.Ra)
end

# Основная логика жизни индивида
@resumable function live(env::ConcurrentSim.Simulation, individual::SIRPerson, m::SIRModel)
    while true
        if individual.status == :S
            contact_time = rand(Exponential(1/m.c))
            death_time = m.μ > 0 ? rand(Exponential(1/m.μ)) : Inf
            wait_time = min(contact_time, death_time)
            @yield timeout(env, wait_time)
            
            if m.μ > 0 && wait_time == death_time
                death_update!(env, m, :S)
                break
            else
                alter = individual
                while alter == individual
                    N = length(m.allIndividuals)
                    index = rand(DiscreteUniform(1, N))
                    alter = m.allIndividuals[index]
                end
                if alter.status == :I && rand(Uniform(0, 1)) < m.β
                    individual.status = :I
                    infection_update!(env, m)
                end
            end
            
        elseif individual.status == :I
            recovery_time = rand(Exponential(1/m.γ))
            death_time = m.μ > 0 ? rand(Exponential(1/m.μ)) : Inf
            wait_time = min(recovery_time, death_time)
            @yield timeout(env, wait_time)
            
            if m.μ > 0 && wait_time == death_time
                death_update!(env, m, :I)
                break 
            else
                individual.status = :R
                recovery_update!(env, m)
            end
            
        else
            if m.μ > 0
                @yield timeout(env, rand(Exponential(1/m.μ)))
                death_update!(env, m, :R)
                break
            else
                @yield timeout(env, Inf)
            end
        end
    end
end

@resumable function birth_process(env::ConcurrentSim.Simulation, m::SIRModel)
    while true
        if m.μ > 0
            @yield timeout(env, rand(Exponential(1/m.μ)))
            new_id = length(m.allIndividuals) + 1
            new_individual = SIRPerson(new_id, :S)
            push!(m.allIndividuals, new_individual)
            birth_update!(env, m)
            @process live(env, new_individual, m)
        else
            @yield timeout(env, Inf)
        end
    end
end

# Функции создания и запуска модели
function MakeSIRModel(u0, p)
    (S, I, R) = u0
    N = S + I + R
    (β, c, γ, μ) = p
    sim = ConcurrentSim.Simulation() # Создаём именно Simulation
    allIndividuals = SIRPerson[]
    for i = 1:S
        push!(allIndividuals, SIRPerson(i, :S))
    end
    for i = (S+1):(S+I)
        push!(allIndividuals, SIRPerson(i, :I))
    end
    for i = (S+I+1):N
        push!(allIndividuals, SIRPerson(i, :R))
    end
    ta = Float64[0.0]
    Sa = Int64[S]
    Ia = Int64[I]
    Ra = Int64[R]
    SIRModel(sim, β, c, γ, μ, ta, Sa, Ia, Ra, allIndividuals)
end

function activate(m::SIRModel)
    for individual in m.allIndividuals
        @process live(m.sim, individual, m)
    end
    @process birth_process(m.sim, m)
end

function sir_run(m::SIRModel, tf::Float64)
    ConcurrentSim.run(m.sim, tf)
end

function out(m::SIRModel)
    result = DataFrame()
    result[!, :t] = m.ta
    result[!, :S] = m.Sa
    result[!, :I] = m.Ia
    result[!, :R] = m.Ra
    return result
end
