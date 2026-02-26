
using Overseer
using Plots

#---------- Component Layer -----------
#

@component struct Arm
    true_mean::Float64 # true reward
end

@component struct ArmID
    id::Int
end

@component mutable struct LearnerState
    Q::Vector{Float64} # estimated values of actions
    N::Vector{Int} # number of times action i taken
end

@component mutable struct RecentAction
    arm_index::Int
    last_reward::Float64
end

# Strategy tags
@component struct EpsilonGreedy
    epsilon::Float64 # exploration generate
end

@component struct UCB
    c::Float64
end

@component struct ThompsonSampling
end

@component mutable struct ActionScores
    scores::Vector{Float64}
end

@component mutable struct History # for plotting later
    actions::Vector{Int} # Store arm_indices over time
    rewards::Vector{Float64}
end


# -------------- System Pipeline ----------------
# This is the logic

struct DecisionSystem <: System end
struct RewardSystem <: System end
struct LearningSystem <: System end
struct RecordingSystem <: System end

function Overseer.update(::RewardSystem, l::AbstractLedger)
    # Get all arms once to avoid nested entity iteration
        # We'll assume arm_index maps to the ArmID component
        arms = @entities_in(l, Arm && ArmID)

        for agent in @entities_in(l, RecentAction)
            idx = agent.arm_index

            # Find the arm with the matching ID
            # In a package, you'd likely cache this mapping for speed
            for arm in arms
                if arm.id == idx
                    agent.last_reward = arm.true_mean + randn()
                    break
            end
        end
    end
end


function Overseer.update(::LearningSystem, l::AbstractLedger)
    # find the correct agent I assume
    for e in @entities_in(l, RecentAction && LearnerState)
        # Read yo memory in RecentAction
        idx = e.arm_index
        r = e.last_reward

        # update N
        e.N[idx] += 1
        n = e.N[idx]

        # Incremental mean update: Q = Q + 1/n * (reward - Q)
        e.Q[idx] += (r - e.Q[idx]) / n
    end
end

function Overseer.update(::DecisionSystem, l::AbstractLedger)
    for e in @entities_in(l, ActionScores && RecentAction)
        # Standard argmax logic
        _, choice = findmax(e.scores)
        e.arm_index = choice
    end
end

function Overseer.update(::RecordingSystem, l::AbstractLedger)
    for e in @entities_in(l, RecentAction && History)
        push!(e.actions, e.arm_index)
        push!(e.rewards, e.last_reward)
    end
end

# strategy systems

struct UCBScorer <: System end
struct EpsilonGreedyScorer <: System end
struct ThompsonSamplingScorer <: System end

function Overseer.update(::UCBScorer, l::AbstractLedger)
    # Only runs for entities tagged with UCB
    for e in @entities_in(l, LearnerState && UCB && ActionScores)
        total_n = sum(e.N)

        for i in eachindex(e.scores)
            if e.N[i] == 0
                # Ensure we try every arm at least once by giving unvisited arms infinite score
                e.scores[i] = Inf
            else
                exploration_bonus = e.c * sqrt(log(total_n) / e.N[i]) # '2.0' is your 'c' parameter
                e.scores[i] = e.Q[i] + exploration_bonus
            end
        end
    end
end

function Overseer.update(::EpsilonGreedyScorer, l::AbstractLedger)
    for e in @entities_in(l, LearnerState && EpsilonGreedy && ActionScores)
        if rand() < e.epsilon
            # Exploration: Assign a random arm an unbeatable score
            # We zero others out or keep Q; Inf ensures it wins the argmax.
            e.scores .= -Inf
            e.scores[rand(1:length(e.scores))] = Inf
        else
            # Exploitation: Just copy current estimates
            e.scores .= e.Q
        end
    end
end

function Overseer.update(::ThompsonSamplingScorer, l::AbstractLedger)
    for e in @entities_in(l, LearnerState && ThompsonSampling && ActionScores)
        for i in eachindex(e.scores)
            if e.N[i]==0
                e.scores[i] = 1e10
            else
                standard_deviation = 1.0/sqrt(e.N[i])
                e.scores[i] = e.Q[i] + (standard_deviation * randn())
            end
        end
    end
end

# ---------- Simulation stage ---------
#

sim_stage = Stage(:execution, [
    UCBScorer(),
    EpsilonGreedyScorer(),
    ThompsonSamplingScorer(),
    DecisionSystem(),
    RewardSystem(),
    LearningSystem(),
    RecordingSystem()
])




# ------------- testbed function -------------

function run_testbed(strategies, T=1000, N=2000)
    # matrix with rows = time and cols = strategies
    n_strats = length(strategies)
    avg_rewards = zeros(T, n_strats) # running average
    optimal_line = zeros(T)
    println("running testbed ...")

    for j in 1:N
        if j % 100 == 0
            println("  > Completed Run $j / $N...")
        end
        # New ledger and new arms -- fresh problem
        l = Ledger(sim_stage)

        # set up new arms
        true_means = randn(10)
        best_mean = maximum(true_means)
        for i in 1:10
            Entity(l, Arm(true_means[i]), ArmID(i))
        end

        # spawn agents with chosen strategies
        agent_ids = []
        for strat in strategies
            # create an agent for each strategy
            id = Entity(l,
                strat,
                LearnerState(zeros(10), zeros(Int, 10)),
                ActionScores(zeros(10)),
                RecentAction(0, 0.0))
            push!(agent_ids, id)
        end

        for t in 1:T
            update(l)

            # keep track of the optimal path that would have had highest payoff
            optimal_line[t] += (best_mean - optimal_line[t]) / j

            for (idx, id) in enumerate(agent_ids)
                # get reward they just got
                r = l[RecentAction][id].last_reward

                # update global average for this specific time step 't' remember j is current run number
                avg_rewards[t, idx] += (r-avg_rewards[t, idx])/ j
            end
        end
    end
    return avg_rewards, optimal_line
end

# Define the "Competitors"
my_strats = [UCB(2.0), EpsilonGreedy(0.1), EpsilonGreedy(0.01), ThompsonSampling()]

# Run the testbed
results, optimal = run_testbed(my_strats)

println("\n" * "="^30)
println("FINAL TESTBED SUMMARY")
println("="^30)

# calculate total number of time steps
total_steps = size(results, 1)

for i in 1:length(my_strats)
    strat_name = string(nameof(typeof(my_strats[i])))
    final_avg = round(results[end, i], digits=3)
    total_cum = round(sum(results[:, i]), digits=2)

    println("Strategy: $strat_name")
    println("  - Final Avg Reward: $final_avg")
    println("  - Total Avg Reward over $total_steps steps: $total_cum")
    println("-"^30)
end

# Plotting logic
p = plot(optimal, label="Optimal (Ceiling)", linestyle=:dash, color=:black)
for i in 1:length(my_strats)
    plot!(p, results[:, i], label=string(nameof(typeof(my_strats[i]))))
end

savefig("bandit_testbed.png")
println("Plot saved to bandit_results.png")
