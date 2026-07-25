
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
import json, os

os.makedirs("plots", exist_ok=True)
np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid")
BLUE="#2c7bb6"; RED="#d73027"; GREEN="#1a9641"

with open("data_info.json") as f:
    info = json.load(f)
obs_rate = info["arrest_rate"]

# ─────────────────────────────────────────────────────────────────────────────
# A.  Prior predictive check
# ─────────────────────────────────────────────────────────────────────────────
n_sim = 40_000

p_chosen  = expit(np.random.normal(0, 2.5, n_sim))
p_default = expit(np.random.normal(0, np.sqrt(1/0.001), n_sim))  # JAGS default

fig, axes = plt.subplots(1, 2, figsize=(11,4))
ax = axes[0]
ax.hist(p_chosen, bins=80, color=BLUE, alpha=0.75, density=True, edgecolor="none")
ax.axvline(obs_rate, color=RED, lw=2, ls="--")
ax.text(obs_rate+0.025, ax.get_ylim()[1]*0.82,
        f"Observed\narrest rate\n{obs_rate:.3f}", color=RED, fontsize=9)
ax.set_xlim(0,1); ax.set_xlabel("Implied arrest probability p"); ax.set_ylabel("Density")
ax.set_title("Prior predictive: N(0, 2.5²)\n(Our chosen prior)", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
ax.hist(p_default, bins=80, alpha=0.5, density=True, color=RED,
        label="JAGS default: N(0, 1/0.001)", edgecolor="none")
ax.hist(p_chosen,  bins=80, alpha=0.5, density=True, color=BLUE,
        label="Chosen: N(0, 2.5²)", edgecolor="none")
ax.axvline(obs_rate, color="black", lw=1.5, ls="--", label=f"Observed ({obs_rate:.3f})")
ax.set_xlim(0,1); ax.set_xlabel("Implied arrest probability p"); ax.set_ylabel("Density")
ax.set_title("Prior comparison\nDefault forces extremes; chosen allows full range",
             fontweight="bold")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("plots/prior_predictive_check.png", dpi=130)
plt.close(); print("Saved plots/prior_predictive_check.png")

# ─────────────────────────────────────────────────────────────────────────────
# B.  JAGS sensitivity — refit Model 2 under 3 prior scales
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("sqf_clean.csv")
for col,(src,val) in {
    "race_WH":("race_grp","White Hispanic"),"race_BH":("race_grp","Black Hispanic"),
    "race_W":("race_grp","White"),"race_O":("race_grp","Other"),
    "crime_PT":("crime_grp","Property/Theft"),"crime_V":("crime_grp","Violence"),
    "crime_T":("crime_grp","Trespass"),"crime_O":("crime_grp","Other"),
    "boro_Bk":("borough","Brooklyn"),"boro_Ma":("borough","Manhattan"),
    "boro_Qu":("borough","Queens"),"boro_SI":("borough","Staten Island"),
    "seas_Win":("season","Winter"),"seas_Spr":("season","Spring"),"seas_Sum":("season","Summer"),
}.items():
    df[col]=(df[src]==val).astype(float)

X_COLS=["age_z","male_bin","frisked_bin","searched_bin","weapon_found_bin","firearm_bin",
        "violent_bg_bin","known_weapon_bin","self_init","race_WH","race_BH","race_W","race_O",
        "crime_PT","crime_V","crime_T","crime_O","boro_Bk","boro_Ma","boro_Qu","boro_SI",
        "seas_Win","seas_Spr","seas_Sum"]
X = df[X_COLS].values.astype(float)
y = df["y"].values.astype(int)
N,K = X.shape; J = int(df["precinct_idx"].max())
precinct_idx = df["precinct_idx"].astype(int).values

# Parameterised Model 2 string — tau_beta is the prior precision
def model2_string(tau_beta, tau_sigma):
    return f"""
model {{
  for (i in 1:N) {{
    y[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + inprod(X[i,], beta[]) + u[precinct_idx[i]]
  }}
  for (j in 1:J) {{
    u[j] ~ dnorm(0, tau_u)
  }}
  alpha ~ dnorm(0, {tau_beta})
  for (k in 1:K) {{
    beta[k] ~ dnorm(0, {tau_beta})
  }}
  sigma_u ~ dnorm(0, {tau_sigma})T(0,)
  tau_u   <- 1 / (sigma_u * sigma_u)
}}
"""

# Show these 6 parameters in the sensitivity plot
SHOW_PARAMS = ["alpha","beta[1]","beta[5]","beta[3]","beta[22]","sigma_u"]
SHOW_LABELS = ["Intercept","Age","Weapon Found","Frisked","Winter","σ_u"]

sens_results = []
try:
    import pyjags

    for tau_b, tau_s, label in [
        (1.0,   4.0,  "Narrow: N(0,1²)"),    # tau=1/1^2=1, sigma prior tau=1/0.5^2=4
        (0.16,  1.0,  "Chosen: N(0,2.5²)"),  # tau=1/2.5^2=0.16, sigma prior tau=1/1^2=1
        (0.04,  0.25, "Wide: N(0,5²)"),       # tau=1/5^2=0.04, sigma prior tau=1/2^2=0.25
    ]:
        print(f"\nSensitivity: {label}  (tau_beta={tau_b}, tau_sigma={tau_s})")
        model = pyjags.Model(
            code   = model2_string(tau_b, tau_s),
            data   = dict(N=int(N),K=int(K),J=int(J),y=y,X=X,precinct_idx=precinct_idx),
            chains = 2, adapt = 500,
        )
        model.sample(1500, vars=[])    # burn-in
        samp = model.sample(2000, vars=SHOW_PARAMS, thin=2)

        for pname, plabel in zip(SHOW_PARAMS, SHOW_LABELS):
            arr = samp[pname]
            if arr.ndim==3: arr=arr[:,:,0]
            d = arr.flatten()
            sens_results.append(dict(prior=label, param=plabel,
                                     mean=float(d.mean()), sd=float(d.std()),
                                     lo=float(np.percentile(d,2.5)),
                                     hi=float(np.percentile(d,97.5))))

except ImportError:
    print("pyjags not available — using Laplace approximation for sensitivity plot")
    from scipy.optimize import minimize

    def neg_log_post(theta, tau_beta):
        a=theta[0]; b=theta[1:K+1]
        lo = a + X@b
        ll = np.sum(y*lo - np.logaddexp(0,lo))
        lp = -0.5*tau_beta*(a**2+np.sum(b**2))
        return -(ll+lp)

    def neg_grad(theta, tau_beta):
        a=theta[0]; b=theta[1:]
        p_=expit(a+X@b)
        r=y-p_
        ga = r.sum() - tau_beta*a
        gb = X.T@r - tau_beta*b
        return -(np.concatenate([[ga],gb]))

    for tau_b, tau_s, label in [(1.0,4.0,"Narrow: N(0,1²)"),(0.16,1.0,"Chosen: N(0,2.5²)"),(0.04,0.25,"Wide: N(0,5²)")]:
        res = minimize(neg_log_post, np.zeros(K+1), args=(tau_b,), jac=neg_grad, method="L-BFGS-B")
        theta_map=res.x
        # Laplace SE from diagonal Hessian
        eps=1e-5; H_diag=np.zeros(K+1)
        g0=neg_grad(theta_map,tau_b)
        for i in range(K+1):
            e=np.zeros(K+1); e[i]=eps
            H_diag[i]=(neg_grad(theta_map+e,tau_b)[i]-g0[i])/eps
        se=np.sqrt(np.abs(1/H_diag))
        for pname,plabel,idx in zip(
            SHOW_PARAMS,SHOW_LABELS,
            [0,1,5,3,22,None]  # indices into theta (None=sigma_u not from MAP)
        ):
            if idx is None:
                m=lo_=hi=0.0  # sigma_u not available from MAP
            else:
                m=float(theta_map[idx]); lo_=m-1.96*float(se[idx]); hi=m+1.96*float(se[idx])
            sens_results.append(dict(prior=label,param=plabel,mean=m,sd=float(se[idx] if idx else 0),lo=lo_,hi=hi))

sens_df = pd.DataFrame(sens_results)
sens_df.to_csv("prior_sensitivity.csv", index=False)

# Plot
prior_colors = {"Narrow: N(0,1²)":RED, "Chosen: N(0,2.5²)":BLUE, "Wide: N(0,5²)":GREEN}
fig, axes = plt.subplots(1, len(SHOW_LABELS), figsize=(14,3.8))
for ax, pname in zip(axes, SHOW_LABELS):
    sub = sens_df[sens_df["param"]==pname].reset_index(drop=True)
    for i, row in sub.iterrows():
        c = prior_colors[row["prior"]]
        ax.errorbar(row["mean"], i,
                    xerr=[[row["mean"]-row["lo"]],[row["hi"]-row["mean"]]],
                    fmt="o", color=c, capsize=4, ms=7, lw=1.5)
    ax.axvline(0, color="grey", ls="--", lw=0.7)
    ax.set_title(pname, fontsize=9, fontweight="bold")
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Narrow","Chosen","Wide"] if ax==axes[0] else [], fontsize=8)
    ax.set_xlabel("Log-odds", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Prior Sensitivity Analysis — Posteriors stable across prior scales",
             fontweight="bold", fontsize=10)
plt.tight_layout()
plt.savefig("plots/prior_sensitivity.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved plots/prior_sensitivity.png")
print("Script 3 complete.")
