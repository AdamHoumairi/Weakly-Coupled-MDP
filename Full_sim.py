import MDP_scaled as mdp
import random
import numpy as np
import pandas as pd
from collections import defaultdict


# -------------------------------------------------
# Global CTMC simulation using the assembled policy
# -------------------------------------------------
def simulate_global(
    lam_cluster,
    adv_vec,
    policy,
    horizon=60000.0,
    warm=10000.0,
    seed=1,
    # TD(0) params
    rho=0.01,                # continuous-time discount rate (>0); gamma = e^{-rho * dt}
    alpha0=0.2,              # initial stepsize
    alpha_cool=1000.0,       # stepsize schedule α_s = α0 / (1 + visits[s]/alpha_cool)
    min_visits_for_eval=10,  # only trust states seen >= this many times
    # verification sampling
):
    rng = random.Random(seed)

    # State
    q_vm = [0]*mdp.N
    q_ct = [[0]*mdp.K for _ in range(mdp.N)]

    # Metrics
    t = 0.0
    last_t = 0.0
    area_L_vm = 0.0
    area_L_ct = 0.0
    arrived = 0
    blocked = 0
    admitted = 0
    area_busy_vm = [0.0] * mdp.N

    # Value tables
    V = defaultdict(float)      # V[key]
    visits = defaultdict(int)   # visit counts for stepsize schedule

    # Initialize current state key
    s_key = mdp.canonical_key(q_vm, q_ct)

    while t < horizon:
        # total rate (arrivals + eligible promotions + services)
        rate = lam_cluster
        for i in range(mdp.N):
            if q_vm[i] > 0 and any(q_ct[i][k] < mdp.C_CT for k in range(mdp.K)):
                rate += mdp.MU_VM
        for i in range(mdp.N):
            for k in range(mdp.K):
                if q_ct[i][k] > 0:
                    rate += mdp.MU_CT
        if rate <= 0.0:
            break

        dt = rng.expovariate(rate)
        t_next = t + dt

        # instantaneous holding (constant between events)
        Ltot = mdp.H_VM*sum(q_vm) + mdp.H_CT*sum(sum(row) for row in q_ct)

        # discounted reward over (t, t+dt): ∫ e^{-ρ(τ-t)} L dτ = L * (1 - e^{-ρ dt}) / ρ
        # block penalty (if any) will be added at event time with an extra e^{-ρ dt} factor
        disc_int = (1.0 - np.exp(-rho*dt)) / max(rho, 1e-12)
        R_cont = Ltot * disc_int

        # Sample event
        rpick = rng.random() * rate
        event = None
        if rpick < lam_cluster:
            event = "arrival"
        else:
            rpick -= lam_cluster
            # promotions
            for i in range(mdp.N):
                if q_vm[i] > 0 and any(q_ct[i][k] < mdp.C_CT for k in range(mdp.K)):
                    if rpick < mdp.MU_VM:
                        # execute promotion at end of interval
                        event = ("promote", i)
                        break
                    rpick -= mdp.MU_VM
            if event is None:
                for i in range(mdp.N):
                    for k in range(mdp.K):
                        if q_ct[i][k] > 0:
                            if rpick < mdp.MU_CT:
                                event = ("service", i, k)
                                break
                            rpick -= mdp.MU_CT
                    if event is not None:
                        break

        # Compute next state by simulating the event
        # We'll also tally steady-state metrics after warm-up
        # TD target needs s' and an impulse reward if block occurs
        # Copy references

        # Discount factor to next state
        gamma = float(np.exp(-rho*dt))
        R_impulse = 0.0

        # Apply event at t_next
        if event == "arrival":
            if t_next > warm:
                arrived += 1
            # choose VM among feasible+locally-ACCEPTING with min advantage
            best_i, best_adv = None, float("inf")
            for i in range(mdp.N):
                s = (q_vm[i], *q_ct[i])
                if mdp.accept_feasible(s):
                    s_idx = mdp.S2I[s]
                    if policy[s_idx] == 1:
                        a = adv_vec[s_idx]
                        if a < best_adv:
                            best_adv, best_i = a, i
            if best_i is None:
                if t_next > warm:
                    blocked += 1
                # impulse block penalty at the boundary, discounted by gamma
                R_impulse += mdp.C_BLOCK
            else:
                if q_vm[best_i] < mdp.C_VM:
                    if t_next > warm:
                        admitted += 1
                    q_vm[best_i] += 1
                else:
                    placed = False
                    for k in range(mdp.K):
                        if q_ct[best_i][k] < mdp.C_CT:
                            q_ct[best_i][k] += 1
                            placed = True
                            break
                    if not placed:
                        if t_next > warm:
                            blocked += 1
                        R_impulse += mdp.C_BLOCK
                    else:
                        if t_next > warm:
                            admitted += 1

        elif event and event[0] == "promote":
            i = event[1]
            if q_vm[i] > 0 and any(q_ct[i][k] < mdp.C_CT for k in range(mdp.K)):
                for k in range(mdp.K):
                    if q_ct[i][k] < mdp.C_CT:
                        q_vm[i] -= 1
                        q_ct[i][k] += 1
                        break

        elif event and event[0] == "service":
            i, k = event[1], event[2]
            if q_ct[i][k] > 0:
                q_ct[i][k] -= 1

        # TD(0) update for the canonical key observed over (t, t+dt]
        s_next_key = mdp.canonical_key(q_vm, q_ct)
        # Total discounted one-step reward = continuous part + discounted impulse at boundary
        R_total = R_cont + gamma * R_impulse
        # Bootstrap target
        target = R_total + gamma * V[s_next_key]
        # Stepsize schedule
        a = alpha0 / (1.0 + visits[s_key] / max(alpha_cool, 1e-9))
        V[s_key] += a * (target - V[s_key])
        visits[s_key] += 1

        # Advance time & accumulate steady-state area after warm-up
        L_vm_inst = sum(q_vm)
        L_ct_inst = sum(sum(row) for row in q_ct)   # if q_ct is [N][K]

        if t_next > warm:
            dt_eff = t_next - max(t, warm)
            if dt_eff > 0:
                area_L_vm += L_vm_inst * dt_eff
                area_L_ct += L_ct_inst * dt_eff
                # util: VM busy if any container in that VM is busy
                for i in range(mdp.N):
                    busy = 1.0 if sum(q_ct[i]) > 0 else 0.0
                    area_busy_vm[i] += busy * dt_eff
        t = t_next
        s_key = s_next_key

    # Steady-state metrics
    duration  = max(horizon - warm, 1e-9)

    mean_L_vm = area_L_vm / duration
    mean_L_ct = area_L_ct / duration

    # admitted is post-warm only (after you fix admitted counting)
    lambda_eff = admitted / duration
    W = float("inf") if lambda_eff <= 0 else ((mean_L_vm + mean_L_ct) / lambda_eff)

    block_prob = blocked / max(arrived, 1)
    block_rate = blocked / duration

    holding_rate = mdp.H_VM * mean_L_vm + mdp.H_CT * mean_L_ct
    avg_cost_rate = holding_rate + mdp.C_BLOCK * block_rate

    util = float(np.mean([x / duration for x in area_busy_vm]))

    perf = {
        "lambda": lam_cluster,
        "policy": "MDP",
        "L_VM": mean_L_vm,
        "L_Cont": mean_L_ct,
        "lambda_eff": lambda_eff,
        "W": W,
        "util": util,
        "block_prob": block_prob,
        "block_rate": block_rate,
        "avg_cost_rate": avg_cost_rate,
    }

    return avg_cost_rate, perf, V, visits
