#=


=#
# component true values a = [randn(10, 0, 1)]
# component actions taken = [10 zeroes]
# component reward = generate from a[i], 1 distribution gaussian
# component expected value of actions - [10 zeroes]
# take an action system
# generate reward system
# update system
#
using Overseer


# gather the parts

@component struct Arm
    true_mean::Float64 # true reward
end

@component mutable struct LearnerState
    Q::Vector{Float64} # estimated values of actions
    N::Vector{Int} # number of times action i taken
end

@component mutable struct RecentAction
    arm_index::Int
    last_reward::Float64
end

@component struct EpsilonGreedy
    epsilon::Float64 # exploration generate
end

@component struct UCB

@component struct ArmID
    id::Int
end

# define system struct
struct ActionSystem <: System end
struct RewardSystem <: System end
struct LearningSystem <: System end

function Overseer.update(::ActionSystem, l::AbstractLedger)
    for e in @entities_in(l, LearnerState && EpsilonGreedy && RecentAction)

        # Explore or exploit?
        if rand() < e.epsilon
            # Explorrre
            choice = rand(1:10)
        else
            # Exploiiiit. Findmax returns value, index. pick arm with highest estimate
            _, choice = findmax(e.Q)
        end
        # get action into the RecentAction component
        e.arm_index = choice
    end
end

function Overseer.update(::RewardSystem, l::AbstractLedger)
    # Find the agent with RecentAction
    for agent in @entities_in(l, RecentAction && LearnerState)
        # what did agent pull?
        target_index = agent.arm_index

        # loop through arms to find the one they pulled
        for machine in @entities_in(l, Arm && ArmID)
            # if its the one generate and update reward
            if machine.id == target_index
                truth = machine.true_mean
                reward = truth + randn()
                agent.last_reward = reward
                break
            end
        end
    end
end


function Overseer.update(::LearningSystem, l::AbstractLedger)
    # find the correct agent I assume
    for agent in @entities_in(l, RecentAction && LearnerState)
        # Read yo memory in RecentAction
        idx = agent.arm_index
        reward = agent.last_reward

        # update N
        agent.N[idx] += 1
        count = agent.N[idx]

        # update Q - belief about true means
        old_val = agent.Q[idx]
        agent.Q[idx] = old_val + (1.0 / count) * (reward - old_val)
    end
end

sim_stage = Stage(:execution, [
    ActionSystem(),
    RewardSystem(),
    LearningSystem()
    ])

# create the world
l = Ledger(sim_stage)

# Assemple the kart
Entity(l,
    LearnerState(zeros(10), zeros(Int, 10)), # start with 0 knowledge
    EpsilonGreedy(0.1), # expolre 10% of the time
    RecentAction(0, 0.0) # memory is nada rn
)

for i in 1:10
    Entity(l,
        Arm(randn()), # true payout
        ArmID(i) # name tag
    )
end

# Loopdyloop
println("somethings a brewin'...")
for t in 1:1000
    update(l)
end
println("done.")

# Get the agent entity (assuming it's the first one with a LearnerState)
agent = first(@entities_in(l, LearnerState))

# Get all the arms to see the truth
arms = @entities_in(l, Arm && ArmID)

println("\nResults:")
println("Arm ID | True Mean | Agent Estimate (Q) | Times Pulled (N)")
println("-"^60)

# We loop through 1 to 10 to print them in order
for i in 1:10
    # Find the specific arm entity for this ID
    # (Note: In a real app we'd map this more efficiently, but this works for viewing)
    arm_entity = first(filter(x -> x.id == i, arms))

    true_val = round(arm_entity.true_mean, digits=3)
    est_val  = round(agent.Q[i], digits=3)
    count    = agent.N[i]

    println("   $i   |   $true_val   |       $est_val      |      $count")
end

# try gemini suggestion for cleaning up reward get rid of inner loop, also figure out how to plot 