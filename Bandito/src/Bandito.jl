module Bandito

# Dependencies and Libraries
using Random

# Environment
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

# Agents
abstract type AbstractBanditAgent end

## Epsilon Greedy
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

## Upper Confidence Bound
mutable struct UCBAgent <: AbstractBanditAgent
    k::Int    # Number of arms
    c::Float64    # How optimistic are we about uncertainty
    Q::Vector{Float64}    # Running estimate of E[ reward of arm i]
    N::Vector{Int}    # Number of times each arm is pulled
    t::Int    # Total time step
end

# Constructor
# This initializes all rewards to 0, 0 arms have been pulled, and time starts at 0
UCBAgent(k::Int=10, c::Float64=1.0) = UCBAgent(k, c, zeros(k), zeros(Int, k), 0)

# For UCB Agent
function select(agent::UCBAgent)
    # Ensure that each arm is tried once
    #untried = findall(==(0), agent.N)
    #if !isempty(untried)
        #return rand(untried)
    #end
    
    # Find all arms that have not been tried twice, randomly pick one
    under_sampled = findall(<(2), agent.N)
    if !isempty(under_sampled)
        return rand(under_sampled)
    end

    # Computing the score recieved
    tt = max(agent.t, 1)
    # UCB Formula
        # Q is our exploitation
        # c  how much we value uncertainty is c is greater, we explore more
        # sqrt(log(t)/N) is our uncertainty
    scores = agent.Q .+ agent.c .* sqrt.(log(tt) ./ agent.N)

    # Choose the arm with the highest upper confidence bound
    # Break ties randomly
    max_score = maximum(scores)
    candidates = findall(==(max_score), scores)
    return rand(candidates)
end

# for UCB Agent (Reward updating!) - Same as before with epsilon_greedy
function update!(agent::UCBAgent, action::Int, reward::Float64)
    agent.N[action] += 1
    step_size = 1.0 / agent.N[action]
    agent.Q[action] += step_size * (reward - agent.Q[action])
end 

# Experiment and Simulation
function run_experiment(agent::AbstractBanditAgent, steps::Int, seed::Union{Nothing, Int}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    # The Bandit environment needs to be created or passed as well, and its k should match agent.k
    env = Bandit(agent.k) # Assuming all AbstractBanditAgent subtypes have a .k field
    optimal_action = argmax(env.q_star)

    # Initialize our vectors of rewards and optimal_picks
    rewards = zeros(Float64, steps)
    optimal_picks = zeros(Float64, steps)

    for t in 1:steps
        # UCB uses time, but previous epsilon_greedy did not
        if hasproperty(agent, :t)
            setfield!(agent, :t, getfield(agent, :t) + 1)
        end
        
        action = select(agent)       # Julia dispatches to the correct select method
        reward = step(env, action)
        update!(agent, action, reward) # Julia dispatches to the correct update! method
        rewards[t] = reward
        if action == optimal_action
            optimal_picks[t] = 1.0
        else
            optimal_picks[t] = 0.0
        end
    end
    return rewards, optimal_picks
end

# This function will run the simulation by passing in a Dict mapping a name to an agent (new fresh agent)
# Then it returns a Dict Name of agent with its rewards adn optimal picks.
function run_simulation(
    agents::Dict{String, Function};
    n_runs::Int,
    steps::Int,
    seed::Union{Nothing, Int}=nothing
)
    results = Dict{String, NamedTuple}()

    for (agent_name, make_agent) in agents
        println("Running simulation for agent: $agent_name ...")

        # Accumulators for averaging
        total_rewards = zeros(Float64, steps)
        total_optimal = zeros(Float64, steps)

        # "Repeating this for 2000 independent runs" [cite: 115]
        for r in 1:n_runs
            # Create a fresh Agent instance for each run , value from our dictionary
            current_agent = make_agent()
            run_seed = seed == nothing ? nothing : seed + r - 1

            rewards, optimal_picks = run_experiment(current_agent, steps, run_seed)
            total_rewards .+= rewards
            total_optimal .+= optimal_picks
        end

        # Calculate averages
        avg_rewards = total_rewards ./ n_runs
        pct_optimal = total_optimal ./ n_runs

        results[agent_name] = (rewards=avg_rewards, optimal=pct_optimal)
    end

    return results
end

# Wrapper for Epsilon_greedy Agent only
function run_simulation(
        epsilon::Float64,
        n_runs::Int,
        steps::Int,
        k_arms::Int = 10,
        seed::Union{Nothing, Int}=nothing
        )
    
    agents = Dict(
        "EpsilonGreedy (epsilon=$(epsilon))" => () -> EpsilonGreedyAgent(k_arms, epsilon)
        )
    
    return run_simulation(agents; n_runs=n_runs, steps=steps, seed=seed)
end

# Wrapper for UCB Agent only
function run_ucb_simulation(
        c::Float64,
        n_runs::Int,
        steps::Int,
        k_arms::Int = 10,
        seed::Union{Nothing, Int}=nothing
        )
    
    agents = Dict(
        "UCB (c=$(c))" => () -> UCBAgent(k_arms, c)
        )
    
    return run_simulation(agents; n_runs=n_runs, steps=steps, seed=seed)
end

# Wrapper to compare Epsilon_greedy with UCB 
function run_comparison(
        epsilon::Float64,
        c::Float64,
        n_runs::Int,
        steps::Int,
        k_arms::Int = 10,
        seed::Union{Nothing, Int}=nothing
        )

    agents = Dict(
        "EpsilonGreedy (epsilon=$(epsilon))" => () -> EpsilonGreedyAgent(k_arms, epsilon),
        "UCB (c=$(c))" => () -> UCBAgent(k_arms, c)
        )

     return run_simulation(agents; n_runs=n_runs, steps=steps, seed=seed)
end

end # module Bandito
