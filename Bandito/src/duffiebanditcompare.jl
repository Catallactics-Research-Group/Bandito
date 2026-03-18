# This code will test if our bandit algorithm can effectively find the OLS Minimum Variance Hedge via Duffie Ch. 7
using Random
using Statistics
using Plots

# Load my bandito code
include("Bandito.jl")
using .Bandito

# First, lets simulate our returns, with a "known" hedge ratio
# spot_return = true_hedge_ratio * futures_return + noise

function simulate_returns(
    num_periods::Int;
    true_hedge_ratio::Float64 = 0.70,
    futures_mean::Float64 = 0.0,
    futures_volatility::Float64 = 0.02,
    noise_volatility::Float64 = 0.01,
    rng::AbstractRNG = Random.default_rng(),
)

    futures_returns = futures_mean .+ futures_volatility .* randn(rng, num_periods)
    noise = noise_volatility .* randn(rng, num_periods)

    spot_returns = true_hedge_ratio .* futures_returns .+ noise

    return spot_returns, futures_returns
end


# Duffie OLS minimum-variance hedge ratio:
#hedge_ratio = Cov(spot_returns, futures_returns) / Var(futures_returns)

function compute_min_variance_hedge_ratio(spot_returns, futures_returns)
    return cov(spot_returns, futures_returns) / var(futures_returns)
end

# Now, second we will use our Bandit Algorithm with UCB
# Reward = -(spot_return - hedge_ratio * futures_return)^2

function run_hedge_ucb_experiment(;
    num_periods::Int = 5000,
    true_hedge_ratio::Float64 = 0.70,
    hedge_ratio_grid = collect(0.0:0.05:1.0),
    exploration_parameter::Float64 = 1.0,
    futures_mean::Float64 = 0.0,
    futures_volatility::Float64 = 0.02,
    noise_volatility::Float64 = 0.01,
    seed::Int = 123,
)

    rng = MersenneTwister(seed)

    # Simulate returns
    spot_returns, futures_returns = simulate_returns(
        num_periods;
        true_hedge_ratio=true_hedge_ratio,
        futures_mean=futures_mean,
        futures_volatility=futures_volatility,
        noise_volatility=noise_volatility,
        rng=rng
    )

    num_actions = length(hedge_ratio_grid)

    agent = Bandito.UCBAgent(num_actions, exploration_parameter)

    chosen_hedge_ratios = Vector{Float64}(undef, num_periods)
    rewards = Vector{Float64}(undef, num_periods)

    select_action(agent) = Bandito.select(agent)

    for t in 1:num_periods

        action_index = select_action(agent)
        hedge_ratio = hedge_ratio_grid[action_index]

        hedged_return = spot_returns[t] - hedge_ratio * futures_returns[t]

        reward = -(hedged_return^2)

        Bandito.update!(agent, action_index, reward)

        chosen_hedge_ratios[t] = hedge_ratio
        rewards[t] = reward
    end

    return chosen_hedge_ratios, rewards, agent, spot_returns, futures_returns, hedge_ratio_grid
end

# Third, how did we do?

function summarize_results(spot_returns, futures_returns, hedge_ratio_grid)

    ols_hedge_ratio = compute_min_variance_hedge_ratio(spot_returns, futures_returns)

    mean_rewards = similar(hedge_ratio_grid, Float64)

    for (i, hedge_ratio) in pairs(hedge_ratio_grid)
        hedged_returns = spot_returns .- hedge_ratio .* futures_returns
        mean_rewards[i] = mean(-(hedged_returns .^ 2))
    end

    best_index = argmax(mean_rewards)
    best_hedge_ratio = hedge_ratio_grid[best_index]

    closest_index = argmin(abs.(hedge_ratio_grid .- ols_hedge_ratio))
    closest_hedge_ratio = hedge_ratio_grid[closest_index]

    return ols_hedge_ratio, best_hedge_ratio, closest_hedge_ratio
end

# Lastly, make some plots to show

function plot_results(chosen_hedge_ratios, rewards;
    true_hedge_ratio=nothing,
    ols_hedge_ratio=nothing)

    p1 = plot(chosen_hedge_ratios,
        xlabel="Time",
        ylabel="Chosen Hedge Ratio",
        title="UCB Learning Hedge Ratio",
        legend=false
    )

    if true_hedge_ratio !== nothing
        hline!(p1, [true_hedge_ratio], linestyle=:dash)
    end

    if ols_hedge_ratio !== nothing
        hline!(p1, [ols_hedge_ratio], linestyle=:dot)
    end

    cumulative_average_reward = cumsum(rewards) ./ (1:length(rewards))

    p2 = plot(cumulative_average_reward,
        xlabel="Time",
        ylabel="Average Reward",
        title="Learning Performance",
        legend=false
    )

    return p1, p2
end

# Run it
function main()

    num_periods = 5000
    true_hedge_ratio = 0.70

    chosen_hedge_ratios, rewards, agent, spot_returns, futures_returns, hedge_ratio_grid =
        run_hedge_ucb_experiment(
            num_periods=num_periods,
            true_hedge_ratio=true_hedge_ratio
        )

    ols_hedge_ratio, best_hedge_ratio, closest_hedge_ratio =
        summarize_results(spot_returns, futures_returns, hedge_ratio_grid)

    learned_hedge_ratio = hedge_ratio_grid[argmax(agent.Q)]

    println("--------------------------------------------------")
    println("True Hedge Ratio:        ", true_hedge_ratio)
    println("OLS Hedge Ratio:         ", ols_hedge_ratio)
    println("Best Grid Hedge Ratio:   ", best_hedge_ratio)
    println("Closest to OLS:          ", closest_hedge_ratio)
    println("Bandit Learned Ratio:    ", learned_hedge_ratio)
    println("--------------------------------------------------")

    p1, p2 = plot_results(
        chosen_hedge_ratios,
        rewards;
        true_hedge_ratio=true_hedge_ratio,
        ols_hedge_ratio=ols_hedge_ratio
    )

    display(p1)
    display(p2)
end

main()