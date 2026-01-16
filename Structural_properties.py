import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("advantage_jsq_data_all_lambda.csv")

# Choose a lambda to analyse (you can loop later)
lam = 4
df_lam = df[df["lambda"] == lam].copy()

# Reconstruct value function: V(s) = min_a Q_a(s)
df_lam["V"] = df_lam[["Q1", "Q2"]].min(axis=1)

violations = []
for s1 in df_lam["S_tot"].unique():
    for s2 in df_lam["S_tot"].unique():
        if s1 < s2:
            V1_max = df_lam[df_lam["S_tot"] == s1]["V"].max()
            V2_min = df_lam[df_lam["S_tot"] == s2]["V"].min()
            if V1_max > V2_min + 1e-8:  # small tolerance
                violations.append((s1, s2, V1_max, V2_min))

print("Monotonicity violations:", len(violations))
for v in violations:
    print(v)


# Aggregate value per total1,total2 (we already have S_tot)
g = df_lam.groupby(["S_tot", "total1", "total2"])["V"].mean().reset_index()

# For convenience, compute imbalance |total1 - total2|
g["imbalance"] = (g["total1"] - g["total2"]).abs()

print(g.head())

def check_schur_convex_for_S(S):
    gS = g[g["S_tot"] == S].sort_values("imbalance")
    print(f"\nS_tot = {S}")
    print(gS[["total1", "total2", "imbalance", "V"]])

    # Simple violation search: if more imbalanced has lower V
    viol = []
    rows = gS.to_dict("records")
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            a = rows[i]
            b = rows[j]
            if b["imbalance"] > a["imbalance"] and b["V"] < a["V"] - 1e-8:
                viol.append((a, b))
    print("Violations for S_tot =", S, ":", len(viol))
    for v in viol:
        print("  more balanced:", (v[0]["total1"], v[0]["total2"]), "V=", v[0]["V"],
              "  more unbalanced:", (v[1]["total1"], v[1]["total2"]), "V=", v[1]["V"])

# Example: check a few S values
for S in sorted(g["S_tot"].unique()):
    check_schur_convex_for_S(S)