
using Overseer

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


    # ---------- Simulation stage ---------
    #

    sim_stage = Stage(:execution, [
        UCBScorer(),
        EpsilonGreedyScorer(),
        DecisionSystem(),
        RewardSystem(),
        LearningSystem(),
        RecordingSystem()
    ])

    l = Ledger(sim_stage)

    for i in 1:10
        Entity(l, Arm(randn()), ArmID(i))
    end

    # The UCB Agent
    Entity(l,
        UCB(2.0),
        LearnerState(zeros(10), zeros(Int, 10)),
        ActionScores(zeros(10)),
        RecentAction(0, 0.0),
        History(Int[], Float64[])
    )

    # The Epsilon-Greedy Agent
    Entity(l,
        EpsilonGreedy(0.1),
        LearnerState(zeros(10), zeros(Int, 10)),
        ActionScores(zeros(10)),
        RecentAction(0, 0.0),
        History(Int[], Float64[])
    )

    println("Running simulation...")
    # official loopdy loop
    for t in 1:1000
        update(l)
    end

    # To visualize:
    for agent in @entities_in(l, History && (UCB || EpsilonGreedy))
        strategy_name = UCB in agent ? "UCB" : "Epsilon-Greedy"
        println("Strategy: $strategy_name | Total Reward: $(sum(agent.rewards))")
    end


    # ------- Plots -------
    using Plots

    # Helper to get cumulative rewards
    function get_cumulative_rewards(ledger)
        p = plot(title="Bandit Strategy Comparison", xlabel="Steps", ylabel="Cumulative Reward")

        for e in @entities_in(ledger, History && (UCB || EpsilonGreedy))
            label = UCB in e ? "UCB (c=$(e.c))" : "EpsilonGreedy (ε=$(e.epsilon))"

            # cumsum turns [1, 2, 3] into [1, 3, 6]
            cumulative_r = cumsum(e.rewards)

            plot!(p, cumulative_r, label=label)
        end
        return p
    end

    # Display the plot
    display(get_cumulative_rewards(l))
    
