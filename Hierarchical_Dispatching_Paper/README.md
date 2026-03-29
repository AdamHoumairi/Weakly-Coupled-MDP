This repository contains the full implementation used in the study:

“Hierarchical Dispatching in VM–Container Systems: Exact MDP Benchmarking and Simulation-Based Diagnostics of JSQ”

The code computes an exact optimal policy for a hierarchical VM–container system using dynamic programming and compares it to the classical Join-the-Shortest-Queue (JSQ) heuristic through structural diagnostics and simulation.

------------------------------------------------------------------------------------------------------------------------------
1. Model Description:
  We consider a two-tier hierarchical service system:
    N = 2 Virtual Machines (VMs)
    Each VM contains:
      A feeder queue (capacity = 1)
      K containers (capacity = 1 each)
    Poisson arrivals with rate λ
    Promotion rate from VM queue to container: μ_vm
    Container service rate: μ_cont
    Discount factor: β
  
  The system is modeled as a finite-state discounted Markov Decision Process (MDP) and solved exactly via policy iteration.

-----------------------------------------------------------------------------------------------------------------------------
2. Main File:
All computations are implemented in:
  MDP_adv_map.py
The script includes:
  Full state-space generation
  Transition probability construction (uniformization)
  Policy iteration for optimal control
  Q-function computation and advantage mapping
  JSQ structural agreement diagnostics
  Event-driven continuous-time simulation
  Performance comparison between JSQ and optimal MDP policy
  Automatic λ-sweep experiments

-------------------------------------------------------------------------------------------------------------------------------
3. Requirements:
Python ≥ 3.9
Install dependencies:
  pip install numpy pandas matplotlib

-------------------------------------------------------------------------------------------------------------------------------
4. How to Run:
From the repository directory:
  python MDP_adv_map.py
The script will:
  Sweep λ from 1 to 10
  Compute the optimal MDP policy for each λ
  Compute JSQ agreement diagnostics
  Run event-driven simulations
  Export CSV files containing all results

-------------------------------------------------------------------------------------------------------------------------------
5. Output Files:
After execution, the following files are generated:
  1-'advantage_jsq_data_all_lambda.csv' Contains:
    Full state description
    Optimal action
    JSQ action
    Q-values
    Advantage values
    Agreement indicator
    λ value
  Used for structural diagnostics and advantage maps.
  2-'jsq_simulation_results.csv' Contains:
    λ
    Blocking probability (JSQ)
    Blocking probability (MDP)
    Average occupancy (JSQ)
    Average occupancy (MDP)
  Used for performance comparison plots.
  3-'jsq_agreement_by_lambda.csv' Contains:
    λ
    Global agreement rate A(λ)
  Used for agreement-rate curves.

-------------------------------------------------------------------------------------------------------------------------------
6. Reproducibility:
Random seeds are fixed for deterministic reproducibility:
  np.random.seed(0)
  random.seed(0)
Exact MDP computations are deterministic.
Simulation results are reproducible given fixed seeds.

-------------------------------------------------------------------------------------------------------------------------------
7. Parameter Configuration:
Key global parameters (modifiable at the top of the script):
  N = 2
  K = 4
  C_vm = 1
  C_cont = 1
  beta = 0.95
  mu_vm = 1.0
  mu_cont = 1.0
  lambda_min = 1
  lambda_max = 10
To reproduce the results in the paper:
  Use N = 2
  Use K = 2 or K = 4
  Sweep λ from 1 to 10

-------------------------------------------------------------------------------------------------------------------------------
8. Computational Notes:
State space size:
    ∣𝑆∣=2^2(1+𝐾)
K = 2 → 64 states
K = 4 → 1024 states
Exact policy iteration remains computationally feasible for these configurations.

-------------------------------------------------------------------------------------------------------------------------------
9. Scientific Transparency
This repository enables full reproducibility of:
  Optimal value functions
  Q-functions
  Advantage landscapes
  Structural agreement diagnostics
  Simulation-based performance metrics
All tables and figures reported in the manuscript can be regenerated from the exported CSV files.

-------------------------------------------------------------------------------------------------------------------------------
10. Citation
If you use this code, please cite:
  Houmairi, A., et al.
  Hierarchical Dispatching in VM–Container Systems: Exact MDP Benchmarking and Simulation-Based Diagnostics of JSQ.
