include("ecs-core.jl")
using Plots

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
