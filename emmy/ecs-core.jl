using Overseer

#---------- Component Layer -----------
#

@component struct Arm end

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
struct LearningSystem <: System end
struct RecordingSystem <: System end

function Overseer.update(::LearningSystem, l::AbstractLedger)
    # find the correct agent I assume
    for e in @entities_in(l, RecentAction && LearnerState)
        # Read yo memory in RecentAction
        idx = e.arm_index
        if idx == 0
            continue
        end
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
        if total_n == 0
            # If no actions have been taken, ensure we try everything
            for i in eachindex(e.scores)
                e.scores[i] = Inf
            end
            continue
        end

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
