module Bandito


using Random


struct Bandit
    k::Int                  # Number of arms (usually 10)
    q_star::Vector{Float64} # True action values
end

function Bandit(k::Int=10)
    # "The true value q*(a) of each of the ten actions was selected
    # according to a normal distribution with mean zero and unit variance"
    q_star = randn(k)
    return Bandit(k, q_star)
end

function step(bandit::Bandit, action::Int)
    # "The actual reward, Rt, was selected from a normal distribution
    # with mean q*(At) and variance 1"
    true_value = bandit.q_star[action]
    reward = true_value + randn()
    return reward
end


abstract type AbstractBanditAgent end

mutable struct EpsilonGreedyAgent <: AbstractBanditAgent
    k::Int
    epsilon::Float64
    Q::Vector{Float64}
    N::Vector{Int}
end

# Constructor
EpsilonGreedyAgent(k::Int=10, epsilon::Float64=0.1) = EpsilonGreedyAgent(k, epsilon, zeros(k), zeros(Int, k))


# For EpsilonGreedyAgent (existing logic)
function select(agent::EpsilonGreedyAgent)
    if rand() < agent.epsilon
        return rand(1:agent.k)
    else
        max_val = maximum(agent.Q)
        candidates = findall(x -> x == max_val, agent.Q)
        return rand(candidates)
    end
end


# For EpsilonGreedyAgent (existing logic)
function update!(agent::EpsilonGreedyAgent, action::Int, reward::Float64)
    agent.N[action] += 1
    step_size = 1.0 / agent.N[action]
    agent.Q[action] += step_size * (reward - agent.Q[action])
end


function run_experiment(agent::AbstractBanditAgent, steps::Int, seed::Union{Nothing, Int}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    # The Bandit environment needs to be created or passed as well, and its k should match agent.k
    env = Bandit(agent.k) # Assuming all AbstractBanditAgent subtypes have a .k field
    optimal_action = argmax(env.q_star)

    rewards = zeros(Float64, steps)
    optimal_picks = zeros(Float64, steps)

    for t in 1:steps
        action = select(agent)       # Julia dispatches to the correct select method
        reward = step(env, action)
        update!(agent, action, reward) # Julia dispatches to the correct update! method
        rewards[t] = reward
        if action == optimal_action
            optimal_picks[t] = 1.0
        end
    end
    return rewards, optimal_picks
end


function run_simulation(
    epsilon::Float64, # Epsilon for EpsilonGreedyAgent
    n_runs::Int,
    steps::Int,
    k_arms::Int = 10,
    seed::Union{Nothing, Int}=nothing
)
    results = Dict{String, NamedTuple}()
    agent_name = "EpsilonGreedy (epsilon=$(epsilon))"

    println("Running simulation for agent: $agent_name ...")

    # Accumulators for averaging
    total_rewards = zeros(Float64, steps)
    total_optimal = zeros(Float64, steps)

    # "Repeating this for 2000 independent runs" [cite: 115]
    for r in 1:n_runs
        # Create a fresh EpsilonGreedyAgent instance for each run
        current_agent = EpsilonGreedyAgent(k_arms, epsilon)

        # Pass the agent instance to run_experiment
        rewards, optimal_picks = run_experiment(
            current_agent,
            steps,
            seed === nothing ? nothing : seed + r - 1
        )

        total_rewards .+= rewards
        total_optimal .+= optimal_picks
    end

    # Calculate averages
    avg_rewards = total_rewards ./ n_runs
    pct_optimal = total_optimal ./ n_runs

    results[agent_name] = (rewards=avg_rewards, optimal=pct_optimal)

    return results
end

end # module Bandito
