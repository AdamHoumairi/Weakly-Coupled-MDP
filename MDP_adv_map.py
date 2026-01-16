import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

# ==========================
# Global parameters (modifiable)
# ==========================

N = 2        # Number of VMs (can be changed later)
K = 4        # Number of containers per VM (can be changed later)
C_vm = 1     # VM queue capacity
C_cont = 1   # Container queue capacity

beta = 0.95      # Discount factor
mu_vm = 1.0      # Service rate at VM
mu_cont = 1.0    # Service rate at containers

# Simulation parameters
simulation_time = 10000
burn_in = 1000
lambda_min = 1
lambda_max = 10

# ==========================
# State space construction
# ==========================

def generate_states(N, K, C_vm, C_cont):
    """
    Generate the full global state space as tuples of occupancies.
    For capacity=1 everywhere, each position is 0 or 1, so state is a bit-vector.
    State ordering:
      (q1_vm, q1_c1, ..., q1_cK, q2_vm, q2_c1, ..., q2_cK, ..., qN_vm, qN_c1, ..., qN_cK)
    """
    per_vm_dim = 1 + K  # VM queue + K containers
    total_positions = N * per_vm_dim

    states = []
    for idx in range(2 ** total_positions):
        bits = [(idx >> b) & 1 for b in range(total_positions)]
        states.append(tuple(bits))
    state_to_index = {s: i for i, s in enumerate(states)}
    return states, state_to_index

# ==========================
# Transition and cost
# ==========================

def Lambda_tot(lambda_rate):
    """
    Total uniformization rate for arrivals.
    For now we simply use lambda_rate directly.
    """
    return lambda_rate

def departure_rate(state, N, K, mu_vm, mu_cont):
    """
    Compute the total departure/promotion rate from the state.
    Each occupied VM queue contributes mu_vm.
    Each occupied container contributes mu_cont.
    """
    rate = 0.0
    per_vm_dim = 1 + K
    for i in range(N):
        base = i * per_vm_dim
        q_vm = state[base]
        q_conts = state[base + 1: base + 1 + K]
        if q_vm == 1:
            rate += mu_vm
        for q in q_conts:
            if q == 1:
                rate += mu_cont
    return rate

def next_states_probs(state, action, states, state_to_index,
                      lambda_rate, N, K, C_vm, C_cont, mu_vm, mu_cont):
    """
    Compute one-step transition probabilities from a given state for a given action.
    Uniformization with arrivals + departures.
    Action is the VM chosen for the incoming job (1..N).
    """
    per_vm_dim = 1 + K
    idx_self = state_to_index[state]
    probs = np.zeros(len(states))

    # Uniformization rate
    Lambda = Lambda_tot(lambda_rate)
    dep_rate = departure_rate(state, N, K, mu_vm, mu_cont)
    Q_rate = Lambda + dep_rate
    if Q_rate == 0:
        probs[idx_self] = 1.0
        return probs

    # Start with self-loop
    probs[idx_self] += 1.0

    # Arrival event (prob Lambda/Q_rate)
    chosen_vm = action  # 1..N
    chosen_vm_idx = chosen_vm - 1
    base = chosen_vm_idx * per_vm_dim
    q_vm = state[base]
    q_conts = list(state[base + 1:base + 1 + K])

    if q_vm < C_vm:
        new_state_list = list(state)
        new_state_list[base] = q_vm + 1
        new_state = tuple(new_state_list)
        probs[state_to_index[new_state]] += Lambda / Q_rate
        probs[idx_self] -= Lambda / Q_rate
    else:
        # VM queue full => blocked; we keep state as is (arrival probability stays on self)
        pass

    # Departure/promotion events (prob dep_rate/Q_rate)
    if dep_rate > 0:
        # Choose uniformly among all occupied queues (VM or containers)
        occ_positions = []
        for i in range(N):
            base = i * per_vm_dim
            if state[base] == 1:
                occ_positions.append(("vm", i))
            for k in range(K):
                if state[base + 1 + k] == 1:
                    occ_positions.append(("cont", i, k))
        if len(occ_positions) > 0:
            prob_each = dep_rate / Q_rate / len(occ_positions)
            for item in occ_positions:
                new_state_list = list(state)
                if item[0] == "vm":
                    i = item[1]
                    base = i * per_vm_dim
                    # VM job attempts to move to a free container; if none, it departs
                    moved = False
                    for k in range(K):
                        if new_state_list[base + 1 + k] < C_cont:
                            new_state_list[base] = 0
                            new_state_list[base + 1 + k] = 1
                            moved = True
                            break
                    if not moved:
                        new_state_list[base] = 0
                else:
                    # container departure
                    _, i, k = item
                    base = i * per_vm_dim
                    new_state_list[base + 1 + k] = 0
                new_state = tuple(new_state_list)
                probs[state_to_index[new_state]] += prob_each
                probs[idx_self] -= prob_each

    probs = np.clip(probs, 0, 1)
    s = probs.sum()
    if s > 0:
        probs /= s
    else:
        probs[idx_self] = 1.0
    return probs

def cost(state, action, N, K, C_vm, C_cont,
         holding_vm=1.0, holding_cont=1.0, block_cost=10.0):
    """
    One-step cost for (state, action).
    Holding cost proportional to number of jobs.
    Additional block cost if chosen VM is full at arrival.
    """
    per_vm_dim = 1 + K
    # Holding cost
    total_jobs_vm = 0
    total_jobs_cont = 0
    for i in range(N):
        base = i * per_vm_dim
        total_jobs_vm += state[base]
        total_jobs_cont += sum(state[base + 1:base + 1 + K])
    cost_holding = holding_vm * total_jobs_vm + holding_cont * total_jobs_cont

    # Blocking cost if chosen VM is full
    chosen_vm = action
    chosen_vm_idx = chosen_vm - 1
    base = chosen_vm_idx * per_vm_dim
    q_vm = state[base]
    q_conts = state[base + 1:base + 1 + K]
    block = 1 if (q_vm >= C_vm and sum(q_conts) >= K * C_cont) else 0

    return cost_holding + block_cost * block

# ==========================
# Policy iteration and advantage
# ==========================

def policy_iteration(states, state_to_index, actions,
                     lambda_rate, N, K, C_vm, C_cont,
                     beta, mu_vm, mu_cont,
                     tol=1e-8, max_iter=1000):
    """
    Policy iteration to compute optimal policy and value function
    for a given lambda_rate.
    Returns:
        policy (array of ints),
        V_opt (value vector),
        Q (dict: action -> Q-values array)
    """
    n_states = len(states)

    # Precompute transition matrices and one-step cost for each action
    P = {a: np.zeros((n_states, n_states)) for a in actions}
    c = {a: np.zeros(n_states) for a in actions}

    for i, s in enumerate(states):
        for a in actions:
            P[a][i, :] = next_states_probs(s, a, states, state_to_index,
                                           lambda_rate, N, K, C_vm, C_cont,
                                           mu_vm, mu_cont)
            c[a][i] = cost(s, a, N, K, C_vm, C_cont)

    # Initialize policy (e.g., always choose action 1)
    policy = np.ones(n_states, dtype=int)

    # Policy iteration
    policy_stable = False
    V = np.zeros(n_states)
    it = 0
    while not policy_stable and it < max_iter:
        it += 1
        # Policy evaluation by iterative method
        while True:
            V_new = np.zeros(n_states)
            for i, s in enumerate(states):
                a = policy[i]
                V_new[i] = c[a][i] + beta * P[a][i, :].dot(V)
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new

        # Policy improvement
        policy_stable = True
        for i, s in enumerate(states):
            action_values = []
            for a in actions:
                q_val = c[a][i] + beta * P[a][i, :].dot(V)
                action_values.append(q_val)
            best_a = actions[int(np.argmin(action_values))]
            if best_a != policy[i]:
                policy_stable = False
                policy[i] = best_a

    V_opt = V.copy()

    # Compute Q-values under optimal policy
    Q = {a: np.zeros(n_states) for a in actions}
    for i, s in enumerate(states):
        for a in actions:
            Q[a][i] = c[a][i] + beta * P[a][i, :].dot(V_opt)

    return policy, V_opt, Q

def build_advantage_dataframe(states, state_to_index, policy, Q, N, K):
    """
    Build a DataFrame with:
    - state components
    - per-VM local backlogs (for VM1 and VM2)
    - total backlog S_tot
    - optimal action vs JSQ action
    - Q-values and advantage between actions
    """
    per_vm_dim = 1 + K
    actions = sorted(Q.keys())
    # For now we assume actions = {1,2} and analyze first two VMs
    assert actions == [1, 2], "This helper assumes actions {1,2} for two VMs."

    rows = []
    for i, s in enumerate(states):
        # VM1
        q1_vm = s[0]
        q1_conts = s[1:1 + K]
        total1 = q1_vm + sum(q1_conts)
        # VM2
        offset2 = per_vm_dim
        q2_vm = s[offset2]
        q2_conts = s[offset2 + 1:offset2 + 1 + K]
        total2 = q2_vm + sum(q2_conts)

        S_tot = total1 + total2  # we ignore extra VMs when computing S_tot here

        # JSQ decision
        if total1 < total2:
            jsq_action = 1
        elif total2 < total1:
            jsq_action = 2
        else:
            jsq_action = 1  # tie-breaking

        rows.append({
            "state": s,
            "q1_vm": q1_vm,
            "q1_c1": q1_conts[0] if K > 0 else 0,
            "q1_c2": q1_conts[1] if K > 1 else 0,
            "q2_vm": q2_vm,
            "q2_c1": q2_conts[0] if K > 0 else 0,
            "q2_c2": q2_conts[1] if K > 1 else 0,
            "total1": total1,
            "total2": total2,
            "S_tot": S_tot,
            "opt_action": policy[i],
            "jsq_action": jsq_action,
            "Q1": Q[1][i],
            "Q2": Q[2][i],
            "adv_1_vs_2": Q[1][i] - Q[2][i]
        })

    df_adv = pd.DataFrame(rows)
    df_adv["jsq_agree"] = (df_adv["opt_action"] == df_adv["jsq_action"])
    return df_adv

def compute_jsq_agreement(df_adv):
    """
    Compute R(S) = P(opt_action == JSQ_action | S_tot = S)
    from the advantage DataFrame.
    We no longer derive tau here; tau is computed globally over lambda.
    """
    R = df_adv.groupby("S_tot")["jsq_agree"].mean().reset_index()
    R.columns = ["S_tot", "R_S"]
    return R

# ==========================
# JSQ simulation
# ==========================

def policy(state):
    """
    JSQ policy for simulation: choose VM with smallest backlog.
    Uses global N and K. Assumes state is ordered as:
      (q1_vm, q1_c1, ..., q1_cK, q2_vm, q2_c1, ..., q2_cK)
    """
    per_vm_dim = 1 + K
    assert N == 2, "policy() currently implemented for N=2 only."

    # VM1
    base1 = 0
    q1_vm = state[base1]
    q1_conts = state[base1 + 1: base1 + 1 + K]
    total1 = q1_vm + sum(q1_conts)

    # VM2
    base2 = per_vm_dim
    q2_vm = state[base2]
    q2_conts = state[base2 + 1: base2 + 1 + K]
    total2 = q2_vm + sum(q2_conts)

    # JSQ with tie-breaking in favour of VM1
    return 1 if total1 <= total2 else 2


def simulate_performance(lambda_rate, N, K, C_vm, C_cont, mu_vm, mu_cont,
                         simulation_time=simulation_time, burn_in=burn_in):
    """
    Continuous-time event-based simulation using JSQ policy for N=2 and arbitrary K.

    State structure:
      For each VM i (i=0,1):
        state[base]     = q_vm (VM waiting buffer, 0 or 1)
        state[base+1: ] = K container occupancies (0/1 each)
      with base = i * (1 + K).

    Events:
      - Arrival at rate lambda_rate:
          choose VM by JSQ, try VM queue, else containers on that VM, else block.
      - Promotion (VM -> container) at rate mu_vm per eligible VM:
          eligible if VM queue occupied and at least one free container on that VM.
      - Service completion at rate mu_cont per busy container.
    """
    assert N == 2, "simulate_performance currently implemented for N=2 only."

    per_vm_dim = 1 + K
    # State initialization: all empty
    state = [0] * (N * per_vm_dim)
    t = 0.0

    blocked_jobs = 0
    arrivals = 0
    queue_lengths = []

    while t < simulation_time:
        # ---- Compute event rates ----
        lambda_arr = lambda_rate

        # Promotion rates (VM -> containers)
        promo_vms = []  # list of VM indices (0 or 1) that can promote
        for i in range(N):
            base = i * per_vm_dim
            q_vm = state[base]
            conts = state[base + 1: base + 1 + K]
            if q_vm > 0 and any(c < C_cont for c in conts):
                promo_vms.append(i)
        promo_rate = mu_vm * len(promo_vms)

        # Service rates (containers)
        busy_containers = []  # list of (vm_index, k_index)
        for i in range(N):
            base = i * per_vm_dim
            for k in range(K):
                if state[base + 1 + k] > 0:
                    busy_containers.append((i, k))
        service_rate = mu_cont * len(busy_containers)

        # Total rate
        R = lambda_arr + promo_rate + service_rate
        if R == 0.0:
            break

        # ---- Sample next event ----
        dt = np.random.exponential(1.0 / R)
        t += dt

        u = np.random.rand()
        if u < lambda_arr / R:
            event = "arrival"
        elif u < (lambda_arr + promo_rate) / R:
            event = "promotion"
        else:
            event = "service"

        # ---- Execute event ----
        if event == "arrival":
            # JSQ policy: action 1 or 2
            a = policy(state)
            vm_idx = a - 1
            base = vm_idx * per_vm_dim

            arrivals += 1

            # Try VM queue first
            if state[base] < C_vm:
                state[base] += 1
            else:
                # Try containers on that VM
                placed = False
                for k in range(K):
                    if state[base + 1 + k] < C_cont:
                        state[base + 1 + k] += 1
                        placed = True
                        break
                if not placed:
                    blocked_jobs += 1

        elif event == "promotion":
            if promo_vms:
                vm_idx = random.choice(promo_vms)
                base = vm_idx * per_vm_dim

                if state[base] > 0:
                    # Move from VM queue to first free container
                    for k in range(K):
                        if state[base + 1 + k] < C_cont:
                            state[base] -= 1
                            state[base + 1 + k] += 1
                            break

        else:  # event == "service"
            if busy_containers:
                vm_idx, k = random.choice(busy_containers)
                base = vm_idx * per_vm_dim
                state[base + 1 + k] = 0

        # ---- Statistics after burn-in ----
        if t > burn_in:
            queue_lengths.append(sum(state))

    blocking_prob = blocked_jobs / arrivals if arrivals > 0 else 0.0
    avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0.0
    return blocking_prob, avg_queue_length



# ==========================
# MDP Simulation
# ==========================

def simulate_performance_mdp(lambda_rate, policy_dict, N, K, C_vm, C_cont, mu_vm, mu_cont,
                             simulation_time=simulation_time, burn_in=burn_in):
    """
    Continuous-time simulation using the MDP optimal policy instead of JSQ.
    policy_dict maps full state tuples
      (q1_vm, q1_c1, ..., q1_cK, q2_vm, q2_c1, ..., q2_cK)
    to actions {1,2}. Currently implemented for N=2 and arbitrary K.
    """
    assert N == 2, "simulate_performance_mdp currently implemented for N=2 only."

    per_vm_dim = 1 + K
    state = [0] * (N * per_vm_dim)
    t = 0.0

    blocked_jobs = 0
    arrivals = 0
    queue_lengths = []

    while t < simulation_time:
        lambda_arr = lambda_rate

        # Promotion
        promo_vms = []
        for i in range(N):
            base = i * per_vm_dim
            q_vm = state[base]
            conts = state[base + 1: base + 1 + K]
            if q_vm > 0 and any(c < C_cont for c in conts):
                promo_vms.append(i)
        promo_rate = mu_vm * len(promo_vms)

        # Service
        busy_containers = []
        for i in range(N):
            base = i * per_vm_dim
            for k in range(K):
                if state[base + 1 + k] > 0:
                    busy_containers.append((i, k))
        service_rate = mu_cont * len(busy_containers)

        R = lambda_arr + promo_rate + service_rate
        if R == 0.0:
            break

        dt = np.random.exponential(1.0 / R)
        t += dt
        u = np.random.rand()

        if u < lambda_arr / R:
            event = "arrival"
        elif u < (lambda_arr + promo_rate) / R:
            event = "promotion"
        else:
            event = "service"

        if event == "arrival":
            arrivals += 1
            st = tuple(state)
            a = policy_dict.get(st, 1)  # default to 1 if missing
            vm_idx = a - 1
            base = vm_idx * per_vm_dim

            if state[base] < C_vm:
                state[base] += 1
            else:
                placed = False
                for k in range(K):
                    if state[base + 1 + k] < C_cont:
                        state[base + 1 + k] += 1
                        placed = True
                        break
                if not placed:
                    blocked_jobs += 1

        elif event == "promotion":
            if promo_vms:
                vm_idx = random.choice(promo_vms)
                base = vm_idx * per_vm_dim
                if state[base] > 0:
                    for k in range(K):
                        if state[base + 1 + k] < C_cont:
                            state[base] -= 1
                            state[base + 1 + k] += 1
                            break

        else:  # service
            if busy_containers:
                vm_idx, k = random.choice(busy_containers)
                base = vm_idx * per_vm_dim
                state[base + 1 + k] = 0

        if t > burn_in:
            queue_lengths.append(sum(state))

    blocking_prob = blocked_jobs / arrivals if arrivals > 0 else 0.0
    avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0.0
    return blocking_prob, avg_queue_length




# ==========================
# Main driver: loop over lambda
# ==========================

def run_for_lambda_sweep():
    """
    For lambda in [lambda_min..lambda_max]:
      - run policy iteration
      - compute advantage dataframe and JSQ agreement per state
      - run JSQ simulation for comparison

    At the end:
      - concatenate all per-lambda data into one CSV
      - compute A(lambda) = mean jsq_agree per lambda
      - compute a single global tau (lambda threshold) for a given gamma.
    """
    states, state_to_index = generate_states(N, K, C_vm, C_cont)
    actions = [1, 2]  # dispatch to VM1 or VM2

    all_adv_dfs = []
    sim_results = []

    for lambda_rate in range(lambda_min, lambda_max + 1):
        print(f"\n=== Lambda = {lambda_rate} ===")
        policy_opt, V_opt, Q = policy_iteration(
            states, state_to_index, actions,
            lambda_rate, N, K, C_vm, C_cont,
            beta, mu_vm, mu_cont
        )

        df_adv = build_advantage_dataframe(states, state_to_index,
                                           policy_opt, Q, N, K)
        df_adv["lambda"] = lambda_rate

        # Optional: R(S) for analysis
        R = compute_jsq_agreement(df_adv)
        print("Mean JSQ agreement for lambda", lambda_rate, "=",
              df_adv["jsq_agree"].mean())

        all_adv_dfs.append(df_adv)

        # Build mapping state -> optimal action for MDP policy
        policy_dict = {states[i]: policy_opt[i] for i in range(len(states))}

        # JSQ simulation for this lambda (toy)
        block_prob_jsq, avg_q_jsq = simulate_performance(
            lambda_rate, N, K, C_vm, C_cont, mu_vm, mu_cont
        )

        # MDP-based simulation for this lambda
        block_prob_mdp, avg_q_mdp = simulate_performance_mdp(
            lambda_rate, policy_dict, N, K, C_vm, C_cont, mu_vm, mu_cont
        )

        sim_results.append({
            "lambda": lambda_rate,
            "blocking_prob_jsq": block_prob_jsq,
            "avg_queue_length_jsq": avg_q_jsq,
            "blocking_prob_mdp": block_prob_mdp,
            "avg_queue_length_mdp": avg_q_mdp
        })


    # Concatenate all lambda results into a single DataFrame
    df_all = pd.concat(all_adv_dfs, ignore_index=True)

    # Save ALL experiments in ONE CSV (per state, per lambda)
    df_all.to_csv("advantage_jsq_data_all_lambda.csv", index=False)

    # JSQ simulation summary per lambda
    df_sim = pd.DataFrame(sim_results)
    df_sim.to_csv("jsq_simulation_results.csv", index=False)
    print("\nJSQ simulation summary:\n", df_sim)

    # Compute A(lambda) = mean jsq_agree per lambda
    jsq_agreement_by_lambda = df_all.groupby("lambda")["jsq_agree"].mean().reset_index()
    jsq_agreement_by_lambda.columns = ["lambda", "A_lambda"]
    jsq_agreement_by_lambda.to_csv("jsq_agreement_by_lambda.csv", index=False)
    print("\nJSQ agreement by lambda:\n", jsq_agreement_by_lambda)

    # Compute a single global tau (lambda threshold)
    gamma = 0.95
    valid = jsq_agreement_by_lambda[jsq_agreement_by_lambda["A_lambda"] >= gamma]
    tau = valid["lambda"].max() if not valid.empty else None
    print(f"\nGlobal tau (lambda threshold) for gamma={gamma} is: {tau}")

    return df_all, jsq_agreement_by_lambda, df_sim, tau

def main():
    np.random.seed(0)
    random.seed(0)

    df_all, jsq_agreement_by_lambda, df_sim, tau = run_for_lambda_sweep()
    print("\nFinal tau (lambda threshold):", tau)

if __name__ == "__main__":
    main()
