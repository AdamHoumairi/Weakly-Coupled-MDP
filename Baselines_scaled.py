import math, random
import numpy as np
import pandas as pd

# ---------------------------
# Model parameters (baseline)
# ---------------------------
N        = 50   # VMs
K        = 2    # containers per VM
C_VM     = 3    # VM queue capacity
C_CT     = 1    # container capacity (binary)
MU_VM    = 2.0  # VM->container promotion rate
MU_CT    = 1.0  # container service rate
H_VM     = 1.0  # holding cost (VM queue)
H_CT     = 1.0  # holding cost (containers)
C_BLOCK  = 15.0 # blocking cost per rejected job

# ---------------------------
# Helpers
# ---------------------------
def vm_has_capacity(qv, q1, q2):
    """Feasible to accept at this VM right now?"""
    return (qv < C_VM) or (q1 < C_CT) or (q2 < C_CT)

def admit_into_vm(qv, q1, q2):
    """Return new (qv,q1,q2) after admission (VM queue preferred, then containers).
       If no capacity, return unchanged and a flag."""
    if qv < C_VM:
        return (qv+1, q1, q2), True
    if q1 < C_CT:
        return (qv, q1+1, q2), True
    if q2 < C_CT:
        return (qv, q1, q2+1), True
    return (qv, q1, q2), False

def pick_vm_jsq(qv, q1, q2):
    """JSQ over VMs that can accept now; tie-break uniformly at random."""
    feas = [i for i in range(N) if vm_has_capacity(qv[i], q1[i], q2[i])]
    if not feas:
        return None
    # total backlog b_i = q_vm + q1 + q2
    b = [qv[i] + q1[i] + q2[i] for i in feas]
    m = min(b)
    cand = [feas[i] for i,bi in enumerate(b) if bi == m]
    return random.choice(cand)

def pick_vm_rr(qv, q1, q2, rr_ptr):
    """RR pointer advances until it finds a feasible VM; if none, return None.
       Returns (index, new_rr_ptr)."""
    start = rr_ptr
    for _ in range(N):
        if vm_has_capacity(qv[rr_ptr], q1[rr_ptr], q2[rr_ptr]):
            idx = rr_ptr
            rr_ptr = (rr_ptr + 1) % N
            return idx, rr_ptr
        rr_ptr = (rr_ptr + 1) % N
        if rr_ptr == start:
            break
    return None, rr_ptr

def pick_vm_pod(qv, q1, q2, d=2):
    """
    Power-of-d choices among VMs with capacity.
    Implementation: sample d VMs uniformly from all VMs, then keep feasible ones.
    If none feasible -> None.
    Tie-break uniformly at random among minima.
    """
    # sample with replacement (standard and cheap); you can change to without replacement if you prefer
    sampled = [random.randrange(N) for _ in range(d)]
    feas = [i for i in sampled if vm_has_capacity(qv[i], q1[i], q2[i])]
    if not feas:
        return None
    b = [qv[i] + q1[i] + q2[i] for i in feas]
    m = min(b)
    cand = [feas[j] for j, bj in enumerate(b) if bj == m]
    return random.choice(cand)


# -----------------------------------------
# Discrete-Event Simulation (one policy)
# -----------------------------------------
def simulate_policy(lam, policy="JSQ", d=2, horizon=40000.0, warm=5000.0, seed=1):
    """
    policy ∈ {"JSQ","RR","POD"}.
    POD uses power-of-d choices with parameter d.
    """
    random.seed(seed)

    # VM state
    q_vm  = [0]*N
    q_ct1 = [0]*N
    q_ct2 = [0]*N
    rr_ptr = 0  # round-robin pointer

    t = 0.0
    last_t = 0.0
    area_L_vm = 0.0
    area_L_ct = 0.0
    area_busy_vm = [0.0]*N  # 'busy' if any container busy

    arrived = 0
    blocked = 0
    admitted = 0

    while t < horizon:
        # Build total event rate
        # Arrival
        rate = lam
        # Promotions only when VM queue>0 and some container idle
        for i in range(N):
            if q_vm[i] > 0 and (q_ct1[i] < C_CT or q_ct2[i] < C_CT):
                rate += MU_VM
        # Container services: each busy container contributes MU_CT
        for i in range(N):
            if q_ct1[i] > 0: rate += MU_CT
            if q_ct2[i] > 0: rate += MU_CT

        if rate <= 0.0:
            break

        # Next event time
        dt = random.expovariate(rate)
        t += dt

        # Statistics accumulation after warm-up
        if t > warm:
            dt_eff = t - max(last_t, warm)
            for i in range(N):
                busy = 1.0 if (q_ct1[i] + q_ct2[i]) > 0 else 0.0
                area_busy_vm[i] += busy * dt_eff
            if dt_eff > 0:
                L_vm = sum(q_vm)
                L_ct = sum(q_ct1) + sum(q_ct2)
                area_L_vm += L_vm * dt_eff
                area_L_ct += L_ct * dt_eff
        last_t = t

        # Select event
        r = random.random() * rate

        # --- Arrival ---
        if r < lam:
            if t > warm:
                arrived += 1
            if policy == "JSQ":
                idx = pick_vm_jsq(q_vm, q_ct1, q_ct2)
            elif policy == "POD":
                idx = pick_vm_pod(q_vm, q_ct1, q_ct2, d=d)
            else:  # RR
                idx, rr_ptr = pick_vm_rr(q_vm, q_ct1, q_ct2, rr_ptr)

            if idx is None:
                if t > warm: blocked += 1
            else:
                (q_vm[idx], q_ct1[idx], q_ct2[idx]), ok = admit_into_vm(q_vm[idx], q_ct1[idx], q_ct2[idx])
                if ok and t > warm:
                    admitted += 1
                elif (not ok) and t > warm:
                    blocked += 1

            continue

        r -= lam

        event_done = False

        # --- Promotions (scan VMs; each eligible contributes MU_VM) ---
        for i in range(N):
            eligible = (q_vm[i] > 0) and (q_ct1[i] < C_CT or q_ct2[i] < C_CT)
            if eligible:
                if r < MU_VM:
                    if q_ct1[i] < C_CT:
                        q_vm[i] -= 1; q_ct1[i] += 1
                    elif q_ct2[i] < C_CT:
                        q_vm[i] -= 1; q_ct2[i] += 1
                    event_done = True
                    break
                r -= MU_VM

        if event_done:
            continue  # IMPORTANT: do not also do a service completion
        # --- Container service completions ---
        for i in range(N):
            if q_ct1[i] > 0:
                if r < MU_CT:
                    q_ct1[i] -= 1
                    event_done = True
                    break
                r -= MU_CT
            if q_ct2[i] > 0:
                if r < MU_CT:
                    q_ct2[i] -= 1
                    event_done = True
                    break
                r -= MU_CT


    # Final metrics
    duration    = max(horizon - warm, 1e-9)
    lam_eff     = admitted / duration
    util_mean   = np.mean([b / duration for b in area_busy_vm])
    mean_L_vm   = area_L_vm / duration
    mean_L_ct   = area_L_ct / duration
    W           = float('inf') if lam_eff <= 0 else ((mean_L_ct + mean_L_vm)/lam_eff)
    holding_rate= H_VM * mean_L_vm + H_CT * mean_L_ct
    block_prob  = blocked / max(arrived, 1)
    block_rate = blocked / duration
    avg_cost_rate = holding_rate + C_BLOCK * block_rate

    return {
        "lambda": lam,
        "policy": policy,
        "L_VM": mean_L_vm,
        "L_Cont": mean_L_ct,
        "lambda_eff": lam_eff,
        "W": W,
        "util": util_mean,
        "block_rate": block_rate,
        "block_prob": block_prob,
        "avg_cost_rate": avg_cost_rate
    }
