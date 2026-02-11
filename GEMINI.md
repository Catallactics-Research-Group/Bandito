# Gemini Project: Bandito

This document provides context for the "Bandito" project, a Julia-based simulation environment for the multi-armed bandit problem.

## Project Overview

**Purpose:** The Bandito project is a scientific computing project written in Julia. Its primary function is to simulate and analyze the performance of different algorithms in a classic multi-armed bandit scenario.

**Core Components:**
- `Bandito.jl`: The main module containing all the project's logic.
- `Bandit`: A struct representing the multi-armed bandit environment itself, with a set number of "arms," each with a true value.
- `EpsilonGreedyAgent`: An agent that implements the epsilon-greedy algorithm to solve the bandit problem. It either explores (chooses a random arm) or exploits (chooses the arm with the highest estimated value) based on the `epsilon` parameter.
- `run_experiment`: A function to run a single experiment for a given agent over a set number of steps.
- `run_simulation`: A higher-level function to run multiple independent experiments and average the results to provide a smoother performance analysis.

**Technology:**
- **Language:** Julia
- **Dependencies:**
    - `Random`: Used for generating random numbers for the bandit's true values, the rewards, and the agent's decisions.

## Building and Running

The project is a standard Julia package.

### 1. Setup Julia Environment

If you haven't already, you need to enter the package manager by typing `]` in the Julia REPL.

### 2. Activate the Project Environment

From within the project's root directory, start the Julia REPL and run:

```julia
] activate .
```

This will create an environment specific to this project.

### 3. Install Dependencies

To install the dependencies listed in `Project.toml` and `Manifest.toml`, run:

```julia
] instantiate
```

### 4. Running a Simulation

You can run a simulation directly from the Julia REPL:

```julia
include("src/Bandito.jl")
using .Bandito

# Run a simulation with epsilon = 0.1 for 2000 runs of 1000 steps each
results = run_simulation(0.1, 2000, 1000)
```

### 5. Testing

There are currently no formal tests in this project. To add tests, you would typically:

1.  Create a `test` directory in the project root.
2.  Add a `runtests.jl` file inside the `test` directory.
3.  Add the `Test` standard library to your project dependencies.
4.  Write your tests in `runtests.jl`.

To run the tests, you would then execute:

```julia
] test
```

## Development Conventions

- **Modularity:** The core logic is encapsulated within the `Bandito` module.
- **Type System:** The code uses Julia's type system to define structs for the `Bandit` and the `EpsilonGreedyAgent`. An abstract type `AbstractBanditAgent` is used to allow for future expansion with other types of agents.
- **Function Naming:** Functions that modify their arguments, like `update!`, are named with a trailing `!`, which is a common Julia convention.
