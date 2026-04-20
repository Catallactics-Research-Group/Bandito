using Distributions  # CHANGE: added for Beta, Normal, InverseGamma
using Statistics     # CHANGE: added for mean()
using Random
using Plots

# ------- Agent's mind -----------
# Beta prior. Beta posterior: at t=0, Beta(1,1). Add u and d counts to alpha and beta as observed
# S_0 is observed in the market at time step t
# simulate 50 time steps. Calculate delta(S_50-S_0). Do this 5000 times. Take the expected value
# Update knowledge about that delta
# Move to time step t+1. \alpha + 1 if S_{t+1} > S_t; else \beta + 1
# Use thompson sampling to choose a new delta
#
# ------- Market ------------
# Simulate market data to feed the agent. Can be historical or simulated, simulated with binomial or otherwise
# Calculate black-scholes delta for comparison?
#
# ------- Limitations --------
# The u and d are adding 1 to alpha and beta, not accounting for nuance in how they are actually calculated
# assuming no early exercise of the option so it can be treated as European


# physics of GBM stock price path
const TRUE_ME = 0.05
const TRUE_SIGMA = 0.20
const DT = 1.0 / 252.0 # 1 time step is 1 trading day

# -----------------------------------------------------------------------
# function for real price data
function see_some_data(spot_prev, DT)
     Z = randn()
     # GBM multiplier
     drift_term = (TRUE_ME - (TRUE_SIGMA^2) / 2.0) * DT
     shock_term = TRUE_SIGMA * sqrt(DT) * Z
     spot_now = spot_prev * exp(drift_term + shock_term)
    return spot_now
end


# -----------------------------------------------------------------------
# function for the posterior sampler
function sim_paths(delta, alpha_spot_dirctn, beta_spot_dirctn, mu_u, sigma_u, mu_d, sigma_d, spot, steps_left, paths, K, call_price)
    spot_0 = spot               # keep initial spot val
    spot_ends = zeros(paths)       # CHANGE: pre-allocate a vector to collect the end-of-path prices

    for i in 1:paths
        spot_i = spot_0             # reset spot_i to spot_0
        draw = rand(Beta(alpha_spot_dirctn, beta_spot_dirctn))  # draw probability of going up or down
        for j in 1:steps_left
            p = rand()            # draw from uniform to randomize up or down in price path
            if p < draw
                u = rand(Normal(mu_u, sigma_u))
                spot_i *= (1 + u)
            else
                d = rand(Normal(mu_d, sigma_d))
                spot_i *= (1 - d)
            end
        end
        spot_ends[i] = spot_i       # store each path's final price in the pre-allocated vector
    end
    stock_pnl = delta .* (spot_ends .- spot_0)
    option_pnl = max.(spot_ends .- K, 0.0) # .- call_price we can exclude since its constant across all deltas, it just shifts the entire utility curve up or down. Doesn't change where peak is.
    payoffs = stock_pnl .- option_pnl
    mean_payoff = mean(payoffs)
    var_payoff = var(payoffs)
    lambda = 0.05 # risk aversion parameter
    utility_val = -var_payoff #mean_payoff - (lambda * var_payoff)

    return utility_val
end

# -----------------------------------------------------------------------
function pick_a_delta(avg_payoffs, alphas_delta, betas_delta, delta_choices)
    # Thompson sampling over the discrete delta choices using the Normal-InverseGamma conjugate:
    #   1. For each delta, sample a variance from InverseGamma(alpha_i, beta_i).
    #   2. Conditioned on that variance, sample a mean payoff from Normal(avg_payoffs[i], sqrt(var)).
    #   3. Pick the delta whose sampled mean is highest.

    var_hat   = [rand(InverseGamma(alphas_delta[i], betas_delta[i])) for i in eachindex(delta_choices)]  # sample variance for each
    mu_hat    = [rand(Normal(avg_payoffs[i], sqrt(var_hat[i])))      for i in eachindex(delta_choices)]  # sample mean conditioned on that variance for each

    delta_idx = argmax(mu_hat)          # argmax returns an index into delta_choices
    delta  = delta_choices[delta_idx]

    # return both the delta value AND its index so update() knows which slot to write to
    return delta, delta_idx
end


# -----------------------------------------------------------------------
function update_delta_beliefs(utility_val, delta_idx, avg_payoffs, alphas_delta, betas_delta, times_selected)

    n = times_selected[delta_idx]   # grab the count before updating; used in all formulas below
    # Normal-InverseGamma conjugate updates for the chosen delta:
    alphas_delta[delta_idx] += 0.5  # alpha grows by 0.5 per observation (standard NIG update)
    if n > 0    # guard: when n==0 the numerator is 0 anyway, but this makes the intent explicit
        betas_delta[delta_idx] += (n * (utility_val - avg_payoffs[delta_idx])^2) / (2 * (n + 1))
    end
    # Update the running mean.
    avg_payoffs[delta_idx] = (n * avg_payoffs[delta_idx] + utility_val) / (n + 1)

    times_selected[delta_idx] += 1  # increment AFTER using n in the formulas above

    return avg_payoffs, alphas_delta, betas_delta, times_selected
end


# -----------------------------------------------------------------------
function update_market_knowledge(spot_now, spot_prev, alpha_spot_dirctn, beta_spot_dirctn, mu_u, sigma_u, mu_d, sigma_d, n_u, n_d)
    if spot_now > spot_prev
        alpha_spot_dirctn  += 1
        # Compute the observed fractional up-return and update mu_u / sigma_u incrementally.
        actual_u  = (spot_now - spot_prev) / spot_prev
        n_u += 1
        old_mu_u  = mu_u
        mu_u  = ((n_u - 1) * old_mu_u + actual_u) / n_u
        if n_u > 1
            sigma_u = sqrt(((n_u - 2) * sigma_u^2 + (actual_u - old_mu_u) * (actual_u - mu_u)) / (n_u - 1))
        end
    else
        beta_spot_dirctn   += 1
        # Compute the observed fractional down-move (positive magnitude) and update mu_d / sigma_d.
        actual_d  = (spot_prev - spot_now) / spot_prev
        n_d      += 1
        old_mu_d  = mu_d
        mu_d      = ((n_d - 1) * old_mu_d + actual_d) / n_d
        if n_d > 1
            sigma_d = sqrt(((n_d - 2) * sigma_d^2 + (actual_d - old_mu_d) * (actual_d - mu_d)) / (n_d - 1))
        end
    end
    # ALTERNATIVE: bundle everything into a NamedTuple for self-documenting field access:
    #   return (alpha_s=alpha_s, beta_s=beta_s, mu_u=mu_u, sigma_u=sigma_u,
    #           mu_d=mu_d, sigma_d=sigma_d, n_u=n_u, n_d=n_d)
    return alpha_spot_dirctn, beta_spot_dirctn, mu_u, sigma_u, mu_d, sigma_d, n_u, n_d
end


# -----------------------------------------------------------------
# Como se dice black-scholes delta
function black_scholes_delta(spot_now, K, steps_left, DT, TRUE_SIGMA, r=0.0)
    # instead of TRUE_SIGMA we could be extra and have it calculate implied volatility or something
    t = DT * steps_left
    if t <= 0.0
        return spot_now > K ? 1.0 : 0.0
    end
    d1 = (log(spot_now / K) + (r+TRUE_SIGMA^2/2) * t)/ (TRUE_SIGMA * sqrt(t))
    standard_normal = Normal(0.0, 1.0)
    bs_delta = cdf(standard_normal, d1)

    return bs_delta
end

function run_simulation()
    # ----- Initialization -----

    # Prior parameters for the Beta distribution over market up-move probability.
    # Per the top comment: alpha counts up moves, beta counts down moves. Start at Beta(1,1) flat prior.
    alpha_s = 1.0
    beta_s = 1.0

    # Initial estimates for the Normal distributions describing up/down move magnitudes.
    # Expressed as fractional returns: u ~ Normal(mu_u, sigma_u), d ~ Normal(mu_d, sigma_d).
    mu_u    = 0.0002    # mean fractional up move  (e.g. 0.01 = 1%)
    sigma_u = 0.0126    # std dev of up moves (roughly 0.20 / sqrt(252))
    mu_d    = 0.0002    # mean fractional down move (positive = price declines)
    sigma_d = 0.0126    # std dev of down moves (roughly 0.20 / sqrt(252))

    # Observation counts used for the incremental mean/variance updates in update_market_knowledge().
    # Separate from alpha_s/beta_s because those carry the Beta prior offset (start at 1, not 0).
    n_u = 0
    n_d = 0

    # Discrete delta choices in [0, 1].
    delta_choices = collect(0.0:0.01:1.0)   # 11 values: 0.0, 0.1, ..., 1.0
    n_deltas      = length(delta_choices)

    # Thompson sampling state: Normal-InverseGamma (NIG) conjugate parameters, one slot per delta.
    avg_payoffs    = zeros(n_deltas)       # running mean payoff (mu_i) for each delta; starts at 0
    alphas_delta   = ones(n_deltas)        # NIG shape  (alpha_i); each starts at 1
    betas_delta   = ones(n_deltas)        # NIG scale  (beta_i);  each starts at 1
    times_selected = zeros(Int, n_deltas)  # how many times each delta has been selected

    spot_prev = 100.0   # initial spot price; used on the very first loop step

    # Simulation constants (match the numbers described in the top comment block)
    N_PATHS     = 5000   # Monte Carlo paths per expectation estimate
    TIME_STEPS = 252    # number of outer market steps to run

    # Agent portfolio
    K = 100
    cash = 5.0
    current_delta = 0.0

    # for plotting, scheming, etc
    agent_deltas = Float64[]
    bs_deltas = Float64[]
    stock_prices = Float64[]
    wealth_history = Float64[]

    # -----------------------------------------------------------------------
    # sim loopdy doop official
    for step in 1:TIME_STEPS   # CHANGE: `for step in 100` -> `for step in 1:N_SIM_STEPS`

        steps_left = TIME_STEPS - step + 1
        push!(stock_prices, spot_prev)

        # pick a delta
        delta_new, delta_idx = pick_a_delta(avg_payoffs, alphas_delta, betas_delta, delta_choices)
        push!(agent_deltas, delta_new)

        # act on the delta, update wealth tracking
        trade_cost = (delta_new - current_delta) * spot_prev
        cash -= trade_cost
        current_delta = delta_new
        naive_wealth = cash + (current_delta * spot_prev) # is the naive_wealth dynamically updating?
        push!(wealth_history, naive_wealth)

        # see some data
        spot_now = see_some_data(spot_prev, DT)

        # update market knowledge and unpack all returned state
        alpha_s, beta_s, mu_u, sigma_u, mu_d, sigma_d, n_u, n_d =
            update_market_knowledge(spot_now, spot_prev, alpha_s, beta_s, mu_u, sigma_u, mu_d, sigma_d, n_u, n_d)

        # how do we feel about our choices
        utility_val = sim_paths(delta_new, alpha_s, beta_s, mu_u, sigma_u, mu_d, sigma_d,spot_now, steps_left, N_PATHS, K, cash)

        avg_payoffs, alphas_delta, betas_delta, times_selected =
            update_delta_beliefs(utility_val, delta_idx, avg_payoffs, alphas_delta, betas_delta, times_selected)

        # get bs delta for comparison later
        bs_delta = black_scholes_delta(spot_now, K, steps_left, DT, TRUE_SIGMA, 0.0)
        push!(bs_deltas, bs_delta)

        spot_prev = spot_now   # switch spot to previous

    end

    final_stock_value = current_delta * spot_prev
    option_payout = max(spot_prev - K, 0.0)
    true_final_wealth = cash + final_stock_value - option_payout

    # plot chosen deltas and bs deltas for comparison
    p = plot(agent_deltas, label="Agent Delta", title="Agent vs Black-Scholes", xlabel="Time Step", ylabel="Delta")
    plot!(p, bs_deltas, label="True BS Delta")

    p_twin = twinx(p)
    plot!(p_twin, stock_prices, label= "Stock Price", color=:green, ylabel="Stock Price (\$)", legend=:bottomright)

    hline!(p_twin, [K], label="Strike Price", color=:green, linestyle=:dash)

    display(p)
    savefig(p, "delta_comparison.png")

    # Output summary metrics to the console
    println("\n--- Simulation End Report ---")
    println("Stock Start: \$100.00 | Stock End: \$", round(spot_prev, digits=2))
    println("Option Liability Paid: -\$", round(option_payout, digits=2))

    naked_final_wealth = 5.0 - option_payout

    println("\n--- Wealth & Hedging ---")
    println("Agent Final Wealth: \$", round(true_final_wealth, digits=2))
    println("Unhedged (Naked) Final Wealth: \$", round(naked_final_wealth, digits=2))
    if true_final_wealth > naked_final_wealth
        println("Agent survived the liability better than a naked position!")
    else
        println("Agent underperformed a naked position.")
    end

    mad_bs = mean(abs.(agent_deltas .- bs_deltas))
    println("\n--- Learning Performance ---")
    println("Average distance from True BS Delta: ", round(mad_bs, digits=4))

    annualized_learned_sigma = sigma_u * sqrt(252)
    println("Agent's final belief of Volatility (Annualized Sigma): ", round(annualized_learned_sigma, digits=4), " (True was ", TRUE_SIGMA, ")")

    # Calculate hedging variance (variance of daily wealth changes is technically different from variance of wealth levels,
    # but let's just look at standard deviation of daily wealth change)
    wealth_changes = diff(wealth_history)
    println("Standard Deviation of Daily Wealth Changes: \$", round(std(wealth_changes), digits=2))

end

run_simulation()


# Read through simulation slow, understand what's going down
# whats a normal inverse gamma prior my guy
# Know your thompson sampling

# Look for sources doing something similar - compare with Gemini Report
# add slide on sources addressing where they are, the gap, my entry point

# Understand tf going on with the slides