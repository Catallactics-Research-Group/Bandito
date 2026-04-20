# ECS-bandit Logic Summary

The `ECS-bandit.jl` script implements a multi-armed bandit simulation using the Entity-Component-System (ECS) architecture via the `Overseer.jl` package.

## Goal and Purpose
The goal of this script is to provide a modular environment to simulate and evaluate various multi-armed bandit algorithms. A **multi-armed bandit model** is a classic reinforcement learning problem where an agent must choose between multiple actions (arms), each with an unknown reward distribution. The challenge is to balance **exploration** (trying different arms to learn their true values) with **exploitation** (choosing the best-known arm to maximize immediate reward).

## What a Single Agent is Doing
A single bandit agent repeatedly selects one of the available "arms" at each time step. After pulling an arm, the agent receives a reward sampled from that arm's underlying distribution. The agent then updates its internal estimate of that arm's value based on the reward received. Over time, the agent uses its specific strategy (e.g., UCB, Epsilon-Greedy, Thompson Sampling) to figure out which arm yields the highest average reward.

## One Loop of the Simulation
In a single loop (one time step) of the simulation, the following sequence occurs for all agents simultaneously:
1. **Scoring:** The agent's strategy evaluates all possible arms and assigns each a "score" based on its current knowledge and exploration rules.
2. **Decision:** The agent selects the arm with the highest score.
3. **Reward:** The environment generates a simulated reward for the chosen arm by adding noise to the arm's true, hidden mean.
4. **Learning:** The agent observes the reward and incrementally updates its estimated value for the chosen arm.
5. **Recording:** The chosen action and the received reward are logged for later analysis.

Here is a high-level breakdown of the overall plan, logic, and output of the script:

## 1. Component Layer (Data)
In the ECS paradigm, data is separated from logic. The script defines several components to represent the state of the simulation:
- **Environment:** `Arm` (stores the true mean reward) and `ArmID` (identifier).
- **Agent State:** `LearnerState` (tracks `Q` estimates and `N` action counts), `RecentAction` (the arm chosen and the reward received), `ActionScores` (computed scores for each arm to determine the next choice), and `History` (for logging).
- **Strategy Tags:** Components used to tag agents with specific behaviors: `EpsilonGreedy`, `UCB`, and `ThompsonSampling`.

## 2. System Pipeline (Logic)
Systems contain the logic that operates on entities with specific components. The pipeline is divided into two main categories:

**Core Simulation Systems:**
- `RewardSystem`: Simulates pulling the arm by adding Gaussian noise to the arm's `true_mean` and assigning it to the agent's `last_reward`.
- `LearningSystem`: Updates the agent's internal estimates (`Q` values and `N` counts) incrementally based on the received reward.
- `DecisionSystem`: Chooses the next arm by simply picking the one with the maximum score (`argmax(scores)`).
- `RecordingSystem`: Logs actions and rewards into the `History` component.

**Strategy Scoring Systems:**
These systems run before the `DecisionSystem` and populate the `ActionScores` array based on the agent's assigned strategy:
- `UCBScorer`: Calculates scores using the Upper Confidence Bound formula (exploitation + exploration bonus). Unvisited arms get infinite score.
- `EpsilonGreedyScorer`: With probability $\epsilon$, assigns a random arm an infinite score (exploration). Otherwise, it assigns the current `Q` estimates as scores (exploitation).
- `ThompsonSamplingScorer`: Samples scores from a normal distribution using the current `Q` value as the mean and $1/\sqrt{N}$ as the standard deviation.

## 3. Simulation Stage
All the systems are bundled together into an execution `Stage` (`sim_stage`). When `update(l)` is called on the ledger, all these systems run in sequence (Scorers -> Decision -> Reward -> Learning -> Recording) to complete one full step of the simulation.

## 4. Testbed Function (`run_testbed`)
This function evaluates the performance of different strategies over many independent runs to get smooth average results.
- **Initialization:** For each of the `N` runs (default 2000), it creates a fresh `Ledger` (a new ECS world), generates 10 new random arms, and spawns one agent for each provided strategy.
- **Simulation Loop:** It runs the simulation for `T` time steps (default 1000). At each step, it calls `update(l)` to advance the simulation and records the reward each agent received.
- **Tracking:** It maintains a running average of the rewards across all `N` runs for each strategy at each time step. It also tracks the "Optimal (Ceiling)" path by always picking the best possible arm for comparison.

## 5. Execution and Output
- **Competitors:** The script evaluates four agents: `UCB(2.0)`, `EpsilonGreedy(0.1)`, `EpsilonGreedy(0.01)`, and `ThompsonSampling()`.
- **Console Output:** It prints a summary to the console showing the final average reward and the total cumulative average reward over the entire run for each strategy.
- **Plotting:** It uses `Plots.jl` to generate a line chart comparing the running average rewards of all strategies against the optimal ceiling over time. The plot is saved to disk as `bandit_results.png` (note: the code saves it to `bandit_testbed.png` but prints that it saved to `bandit_results.png`).

Notes: do it in an economy - mental simulation do at 5,000, we want to say in the limit it approaches black scholes. Moving into the catallaxy, we don't assume a communism of model - model structure isn't assumed. 

In catallaxy - don't use binomial model to simulate price data. show binomial pricing model still predictive, give agent another sampler or simulate multiple sampler outcomes, but its more bayesian now bc not assuming a communism of models. 

Cite testfastion in paper - dynamic games, games in extensive forms agent based modeling., cite buchanan's game theory articles.  

We want catallactic theory and methodology of agent-based computation. 

assuming opponent will take the optimal option; informed trader is implicit in the sampler modeling. 

Reread fig. what is bounded rationality. 