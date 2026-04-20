# 1. Setup the "True" World (Binomial Tree with known optimal delta at each node)
# 2. Setup Agent's Portfolio: Short 1 Call Option
# 3. Specify action space: discrete hedge ratios [0.0, 0.05, 0.10 ... 1.0]
# 4. Specify decision strategy (UCB, Thompson, EpsilonGreedy)
# 5. Define Utility Function: Asymmetric loss based on portfolio wealth
#
# for t in T (or traversing the true binomial tree):
#   -- Agent's Internal Decision Process --
#   1. Agent uses strategy (e.g., UCB) to select an arm (a specific hedge ratio)
#   2. Agent uses their predictive sampler to simulate *one* path forward
#   3. Calculate hypothetical portfolio wealth at the end of that simulated path
#   4. Pass wealth through asymmetric loss function to get a 'simulated reward'
#
#   -- Learning & Updating --
#   5. Update the Q-value (expected reward) of the chosen arm based on the simulated reward
#
#   -- Evaluation (The "True" Step) --
#   6. Compare the Agent's chosen hedge ratio to the known mathematically optimal Delta for time `t`.
#   7. Record if the agent is learning to pick the right arm.
# end
#

using Distributions
using Plots
include("ecs-core.jl")

# --- Mathematics of Delta Hedging ---
# Delta represents the rate of change of the option's price with respect to the underlying asset's price.
# Because both the stock price (S) and time to maturity (t) change continuously, the optimal hedge ratio
# (Delta) is NOT static! It is dynamic. This is why the "True Delta" jumps around.
# As an option gets closer to expiration (T), if it's "in-the-money" (S > K), Delta approaches 1.0.
# If it's "out-of-the-money" (S < K), Delta approaches 0.0.
function true_binomial_delta(S, t, T, K, u, d, r)
    # If the option has expired, the delta is 1 if we are in the money, or 0 if out of the money.
    if t >= T
        return S >= K ? 1.0 : 0.0
    end

    # Risk-neutral probability measure (q) used for pricing, not simulating real life.
    q = (exp(r) - d) / (u - d)

    # A helper function that calculates the Black-Scholes/Binomial price of an option
    # at a specific node in the tree using backward induction.
    function price(S_val, current_t)
        steps_left = T - current_t
        if steps_left == 0
            return max(S_val - K, 0.0)
        end
        prices = zeros(steps_left + 1)
        for i in 0:steps_left
            S_end = S_val * (u^i) * (d^(steps_left - i))
            prices[i+1] = max(S_end - K, 0.0)
        end
        for step in steps_left:-1:1
            for i in 0:(step-1)
                prices[i+1] = exp(-r) * (q * prices[i+2] + (1 - q) * prices[i+1])
            end
        end
        return prices[1]
    end

    # Delta = (Price_up - Price_down) / (Stock_up - Stock_down)
    S_u = S * u
    S_d = S * d
    C_u = price(S_u, t + 1)
    C_d = price(S_d, t + 1)

    return (C_u - C_d) / (S_u - S_d)
end

# ----- Make some special option market maker components -----

# Components are just pure data structs. They hold properties but zero logic.

# The rules of the universe our simulation lives in
@component struct BinomialParameters
    u::Float64 # Up move factor
    d::Float64 # Down move factor
    p::Float64 # The TRUE probability of an up move (hidden from agent)
    r::Float64 # Risk-free interest rate
end

# The exact moment in time and the current stock price
@component mutable struct MarketState
    S::Float64 # Current underlying stock price
    t::Int     # Current time step
end

# Details of the financial contract being traded
@component struct CallOption
    K::Float64 # Strike price
    T::Int     # Total time steps until option expiration
end

# The agent's bank account & holdings
@component mutable struct Portfolio
    cash::Float64
    shares::Float64 # The quantity of stock the agent holds (this is the Delta they chose!)
    options::Int    # Will be fixed at -1 because the agent sold (shorted) one call
end

# A label for a specific discrete action (e.g. "I hold 0.5 shares")
@component struct HedgeRatio
    delta::Float64
end

# The agent's mental model of the world's uncertainty
@component mutable struct BeliefState
    alpha::Int       # Times it saw the market go UP
    beta::Int        # Times it saw the market go DOWN
    estimated_u::Float64 # The agent's guess at the up factor
    estimated_d::Float64 # The agent's guess at the down factor
end

# A broadcast message telling the agent what just happened in the real market
@component mutable struct LastMarketMove
    direction::Symbol # :UP, :DOWN, or :NONE
    old_S::Float64
    new_S::Float64
end

# --- Utility Function ---
# Market makers hate losing money more than they love making it.
# This penalizes negative wealth severely.
function asymmetric_utility(final_wealth::Float64, penalty_multiplier::Float64 = 5.0)
    return final_wealth >= 0.0 ? final_wealth : (final_wealth * penalty_multiplier)
end


# --- NEW SYSTEMS TO IMPLEMENT ---
# Systems are pure logic. They contain no state. They iterate over entities
# that have specific components and modify their data.

struct MentalSimulationSystem <: System end
struct RealExecutionSystem <: System end
struct RealMarketStepSystem <: System end
struct BeliefUpdateSystem <: System end


# The "Imagination" System where the agent simulates the future
function Overseer.update(::MentalSimulationSystem, l::AbstractLedger)
    # --- FIND THE WORLD ---
    # We need to find the specific "World" entity that holds the MarketState and CallOption.
    # In ECS (Entity Component System), we don't have a direct variable for "the world".
    # Instead, we query the ledger (l) for any entity that has BOTH a MarketState and a CallOption component.
    S_start = 0.0
    t_start = 0
    T_val = 0
    K_val = 0.0
    found_world = false

    # The @entities_in macro returns an iterator of entities matching our component requirements.
    # We iterate over them (even though there is only one "World" entity in our simulation).
    for w in @entities_in(l, MarketState && CallOption)
        S_start = w.S # Grab the current real-world stock price
        t_start = w.t # Grab the current real-world time step
        T_val = w.T   # Grab the option's maturity time
        K_val = w.K   # Grab the option's strike price
        found_world = true
        break # We only need the first matching entity, so we break out of the loop
    end
    if !found_world return end # Safety check: If no world exists, exit the function.

    # --- GET ALL THE ARMS ---
    # We get a list of all entities that represent actions (Arms) so we can look up their values later
    arms = @entities_in(l, ArmID && HedgeRatio)

    # --- SIMULATE FOR EACH AGENT ---
    # Find any entity that is an Agent (has RecentAction and BeliefState)
    for agent in @entities_in(l, RecentAction && BeliefState)
        chosen_id = agent.arm_index # The UCB scorer just picked this arm ID for us to test!

        # Find the specific Delta value associated with the chosen Arm ID
        chosen_delta = 0.0
        for arm in arms
            if arm.id == chosen_id
                chosen_delta = arm.delta
                break
            end
        end

        # Reset our imaginary variables to the real world's current state
        S = S_start
        t = t_start
        T = T_val
        K = K_val

        # If the option is already expired, there's no reward to be had.
        if t >= T
            agent.last_reward = 0.0
            continue
        end

        # --- THOMPSON SAMPLING FOR BELIEFS ---
        # The agent isn't perfectly confident in its p.
        # It rolls the dice based on its past observations (alpha and beta) to sample a 'p'
        # that it will use for THIS specific mental simulation.
        p_est = rand(Beta(agent.alpha, agent.beta))
        u_est = agent.estimated_u
        d_est = agent.estimated_d

        # --- THE FORWARD PATH ---
        # Simulate ONE imaginary path forward all the way to maturity (T)
        while t < T
            if rand() < p_est
                S = S * u_est # Stock went up!
            else
                S = S * d_est # Stock went down!
            end
            t += 1
        end

        # --- CALCULATE THE CONSEQUENCES ---
        # The path has reached maturity. Let's see how our chosen delta hedge performed.

        # How much money did buying 'chosen_delta' shares make/lose over the whole path?
        hedge_pnl = chosen_delta * (S - S_start)

        # How much money do we OWE the person who bought the call option from us?
        option_liability = max(S - K, 0.0)

        # Our final wealth is the profit from our hedge MINUS what we owe.
        # If we hedged perfectly, final_wealth will be exactly 0!
        final_wealth = hedge_pnl - option_liability

        # Pass it through our utility function to get the actual reward (loss is penalized heavily)
        reward = asymmetric_utility(final_wealth)

        # Save the reward so the LearningSystem can update the Q-values later
        agent.last_reward = reward
    end
end

# System that locks in the agent's best guess for the real world
function Overseer.update(::RealExecutionSystem, l::AbstractLedger)
    arms = @entities_in(l, ArmID && HedgeRatio)
    for agent in @entities_in(l, LearnerState && Portfolio)
        # Look at the Q array (expected rewards). Find the index of the highest value.
        best_id = argmax(agent.Q)

        # Set our actual portfolio shares to that Delta
        for arm in arms
            if arm.id == best_id
                agent.shares = arm.delta
                break
            end
        end
    end
end

# System that progresses time forward one step
function Overseer.update(::RealMarketStepSystem, l::AbstractLedger)
    for w in @entities_in(l, MarketState && BinomialParameters && LastMarketMove)
        old_S = w.S
        # Flip the TRUE coin (hidden from agent)
        if rand() < w.p
            new_S = old_S * w.u
            w.direction = :UP
        else
            new_S = old_S * w.d
            w.direction = :DOWN
        end
        # Update the real world
        w.S = new_S
        w.t += 1

        # Broadcast what just happened so the agent can learn
        w.old_S = old_S
        w.new_S = new_S
    end
end

# System that allows the agent to observe the real market and update its Beta distribution
function Overseer.update(::BeliefUpdateSystem, l::AbstractLedger)
    direction = :NONE
    # Read the broadcast message from the world
    for w in @entities_in(l, LastMarketMove)
        direction = w.direction
        break
    end
    if direction == :NONE return end

    for agent in @entities_in(l, BeliefState)
        # Bayesian updating!
        if direction == :UP
            agent.alpha += 1 # We saw an up move, get more confident about up moves!
        elseif direction == :DOWN
            agent.beta += 1  # We saw a down move, get more confident about down moves!
        end
    end
end

# --- STAGE SETUP ---

# Stage 1: The Agent sits in a room and hallucinates 50 different futures to plan its next move
mental_stage = Stage(:mental_simulation, [
    UCBScorer(),
    EpsilonGreedyScorer(),
    ThompsonSamplingScorer(),
    DecisionSystem(),
    MentalSimulationSystem(),
    LearningSystem()
])

# Stage 2: The Agent actually places a trade, and the market moves 1 tick
real_world_stage = Stage(:real_world, [
    RealExecutionSystem(),
    RealMarketStepSystem(),
    BeliefUpdateSystem(),
    RecordingSystem()
])

# ---------- Initialize ledger & entities -----------
l = Ledger()

# setup market parameters
u_factor = 1.05
d_factor = 1.0 / u_factor # assuming d = 1/u
prob_u = 0.5
total_steps = 50
init_stock_price = 100.0
strike = 100.0
option_price = 5.0 # assuming option was sold for $5, this is initial premium agent will recieve

# create world entity
world_entity = Entity(l,
    BinomialParameters(u_factor, d_factor, prob_u, 0.0),
    MarketState(init_stock_price, 0),
    CallOption(strike, total_steps),
    LastMarketMove(:NONE, init_stock_price, init_stock_price)
)

# action space: hedge ratios
hedge_ratios = collect(0.0:0.05:1.0)
num_arms = length(hedge_ratios)

# create agent entity
agent_entity = Entity(l,
    Portfolio(option_price, 0.0, -1),
    LearnerState(zeros(num_arms), zeros(Int, num_arms)),
    ActionScores(zeros(num_arms)),
    RecentAction(1, 0.0),
    UCB(2.0),
    BeliefState(1, 1, u_factor, d_factor),
    History(Int[], Float64[])
)

for (i, delta_val) in enumerate(hedge_ratios)
    Entity(l, Arm(), ArmID(i), HedgeRatio(delta_val))
end


# --- Execution Loop ---
total_real_steps = total_steps
mental_sims_per_step = 50 # Compute budget for bounded rationality

agent_chosen_deltas = Float64[]
true_optimal_deltas = Float64[]
true_stock_prices = Float64[]
time_steps = Int[]

for t in 1:total_real_steps

    # Record true optimal delta for current state BEFORE taking a step
    current_S = 0.0
    for w in @entities_in(l, MarketState)
        current_S = w.S
        break
    end

    true_delta = true_binomial_delta(current_S, t-1, total_steps, strike, u_factor, d_factor, 0.0)
    push!(true_optimal_deltas, true_delta)
    push!(time_steps, t-1)
    push!(true_stock_prices, current_S)

    # --- THE FIX: FORGETTING THE PAST TO PLAN FOR THE PRESENT ---
    # WHY THE AGENT WAS STUCK: Before, we never reset `agent.Q`. This meant the agent was learning
    # one single "average" Delta for the ENTIRE 10-step simulation.
    # But Delta is dynamic! It changes based on the CURRENT stock price and time left.
    # To act rationally at time `t`, the agent must "clear its head" and run a fresh set of 50
    # mental simulations specifically tailored to solving the problem from the *current* state.
    for agent in @entities_in(l, LearnerState)
        fill!(agent.Q, 0.0)
        fill!(agent.N, 0)
    end
    for agent in @entities_in(l, ActionScores)
        fill!(agent.scores, 0.0)
    end

    # 1. The Agent "Thinks" (Inner Loop)
    for _ in 1:mental_sims_per_step
        Overseer.update(mental_stage, l)
    end

    # 2. The Agent Acts and the World Moves (Outer Loop)
    Overseer.update(real_world_stage, l)

    # Record agent's chosen delta
    agent_delta = 0.0
    for a in @entities_in(l, Portfolio)
        agent_delta = a.shares
        break
    end
    push!(agent_chosen_deltas, agent_delta)

end

# --- Output and Visualization ---
println("\n=== SIMULATION SUMMARY ===")
println("Total time steps: ", total_steps)
println("Mental simulations per step: ", mental_sims_per_step)
println("\nAgent Beliefs at End:")
for a in @entities_in(l, BeliefState)
    println("Alpha: ", a.alpha, ", Beta: ", a.beta, " -> Estimated p_up = ", round(a.alpha / (a.alpha + a.beta), digits=4))
    break
end

# We don't print global Q values anymore because the Agent wipes its Q memory at every time step!
# Instead, we just show the final results in the plot.

# Set explicit limits so the two axes perfectly match and don't spill over
min_price = min(minimum(true_stock_prices), strike)
max_price = max(maximum(true_stock_prices), strike)
price_padding = (max_price - min_price) * 0.1
price_padding = price_padding == 0.0 ? 5.0 : price_padding # prevent 0 padding

# Create empty space at the bottom 25% of the graph for the legend
delta_top = 1.05
delta_data_bottom = -0.05
delta_bottom = delta_top - ((delta_top - delta_data_bottom) / 0.75)

stock_top = max_price + price_padding
stock_data_bottom = min_price - price_padding
stock_bottom = stock_top - ((stock_top - stock_data_bottom) / 0.75)

# Plotting the deltas
plt = plot(time_steps, true_optimal_deltas, label="True Optimal Delta", marker=:circle, lw=2,
           legend=:bottom, legendcolumns=2, color=:blue,
           title="Agent vs True Delta & Stock Price", xlabel="Time Step", ylabel="Delta Ratio",
           ylims=(delta_bottom, delta_top), xlims=(time_steps[1], time_steps[end]), framestyle=:box)

plot!(plt, time_steps, agent_chosen_deltas, label="Agent Chosen Delta", marker=:square, lw=2, linestyle=:dash, color=:orange)

# Dummy series so all legends are neatly combined in the primary axis legend box
plot!(plt, [NaN], [NaN], label="Stock Price", color=:green, lw=2, linestyle=:dot)
plot!(plt, [NaN], [NaN], label="Strike Price", color=:green, lw=2, linestyle=:solid)

# Add a secondary y-axis for the stock price
plt2 = twinx(plt)
plot!(plt2, time_steps, true_stock_prices, label="", color=:green, lw=2, linestyle=:dot, legend=false,
      ylims=(stock_bottom, stock_top), xlims=(time_steps[1], time_steps[end]))
# Use a normal line segment instead of hline! to avoid breaking x-axis limits
plot!(plt2, [time_steps[1], time_steps[end]], [strike, strike], label="", color=:green, lw=2, linestyle=:solid)
ylabel!(plt2, "Stock Price (\$)")
savefig(plt, "emmy/delta_comparison.png")
println("\nPlot saved to emmy/delta_comparison.png")
