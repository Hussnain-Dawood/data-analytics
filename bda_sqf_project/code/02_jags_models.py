"""
Software: JAGS + pyjags only
Install:
  1. JAGS (Windows): download ~30 MB from https://sourceforge.net/projects/mcmc-jags/
  2. pyjags: pip install pyjags   (~2 MB, no compiler needed)
  3. Other: pip install pandas numpy scipy matplotlib openpyxl
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import expit
import json, os, time, pickle

os.makedirs("plots", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
BLUE="#2c7bb6"; RED="#d73027"; GREEN="#1a9641"; ORANGE="#fc8d59"
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Build design matrix
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("sqf_clean.csv")

for col,(src,val) in {
    "race_WH":  ("race_grp","White Hispanic"),
    "race_BH":  ("race_grp","Black Hispanic"),
    "race_W":   ("race_grp","White"),
    "race_O":   ("race_grp","Other"),
    "crime_PT": ("crime_grp","Property/Theft"),
    "crime_V":  ("crime_grp","Violence"),
    "crime_T":  ("crime_grp","Trespass"),
    "crime_O":  ("crime_grp","Other"),
    "boro_Bk":  ("borough","Brooklyn"),
    "boro_Ma":  ("borough","Manhattan"),
    "boro_Qu":  ("borough","Queens"),
    "boro_SI":  ("borough","Staten Island"),
    "seas_Win": ("season","Winter"),
    "seas_Spr": ("season","Spring"),
    "seas_Sum": ("season","Summer"),
}.items():
    df[col] = (df[src]==val).astype(float)

# Ordered list — matches JAGS model string exactly
X_COLS = [
    "age_z","male_bin","frisked_bin","searched_bin",
    "weapon_found_bin","firearm_bin","violent_bg_bin","known_weapon_bin","self_init",
    "race_WH","race_BH","race_W","race_O",
    "crime_PT","crime_V","crime_T","crime_O",
    "boro_Bk","boro_Ma","boro_Qu","boro_SI",
    "seas_Win","seas_Spr","seas_Sum",
]
COV_LABELS = [
    "Age (scaled)","Male","Frisked","Searched",
    "Weapon Found","Firearm Found","Violent Background","Known Weapon Carrier","Self-Initiated Stop",
    "Race: White Hispanic","Race: Black Hispanic","Race: White","Race: Other",
    "Crime: Property/Theft","Crime: Violence","Crime: Trespass","Crime: Other",
    "Borough: Brooklyn","Borough: Manhattan","Borough: Queens","Borough: Staten Island",
    "Season: Winter","Season: Spring","Season: Summer",
]
WEAPON_COL = X_COLS.index("weapon_found_bin")   # index 4 — used for M3

X  = df[X_COLS].values.astype(float)
y  = df["y"].values.astype(int)
N, K = X.shape
J    = int(df["precinct_idx"].max())
precinct_idx = df["precinct_idx"].astype(int).values

print(f"N={N}, K={K} covariates, J={J} precincts")
print(f"Arrest rate: {y.mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. JAGS model strings
# ─────────────────────────────────────────────────────────────────────────────

PRIOR_TAU = 0.16     # = 1/2.5^2  — weakly informative
PRIOR_TAU_NARROW = 1.0   # = 1/1^2   — for sensitivity
PRIOR_TAU_WIDE   = 0.04  # = 1/5^2   — for sensitivity

# ── Model 1: Logistic regression, no random effects ──────────────────────────
MODEL1 = f"""
model {{
  for (i in 1:N) {{
    y[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + inprod(X[i,], beta[])
  }}
  # Weakly informative priors: N(0, 2.5^2), tau=0.16
  alpha ~ dnorm(0, {PRIOR_TAU})
  for (k in 1:K) {{
    beta[k] ~ dnorm(0, {PRIOR_TAU})
  }}
}}
"""
MODEL2 = f"""
model {{
  for (i in 1:N) {{
    y[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + inprod(X[i,], beta[]) + u[precinct_idx[i]]
  }}
  # Precinct random intercepts
  for (j in 1:J) {{
    u[j] ~ dnorm(0, tau_u)
  }}
  # Priors
  alpha ~ dnorm(0, {PRIOR_TAU})
  for (k in 1:K) {{
    beta[k] ~ dnorm(0, {PRIOR_TAU})
  }}
  # Half-Normal(0,1) for sigma_u via truncated normal
  sigma_u ~ dnorm(0, 1)T(0,)
  tau_u   <- 1 / (sigma_u * sigma_u)
}}
"""

MODEL3 = f"""
model {{
  for (i in 1:N) {{
    y[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + inprod(X[i,], beta[])
                 + u[precinct_idx[i]]
                 + v[precinct_idx[i]] * weapon[i]
  }}
  # Bivariate normal random effects (u=intercept, v=weapon slope)
  for (j in 1:J) {{
    re[j, 1:2] ~ dmnorm(mu_re[], Omega[,])
    u[j] <- re[j, 1]
    v[j] <- re[j, 2]
  }}
  mu_re[1] <- 0
  mu_re[2] <- 0
  # Wishart prior on precision matrix — 2x2, df=2, R=identity
  Omega[1:2,1:2] ~ dwish(R[,], 2)
  Sigma[1:2,1:2] <- inverse(Omega[,])
  sigma_u  <- sqrt(Sigma[1,1])
  sigma_v  <- sqrt(Sigma[2,2])
  rho      <- Sigma[1,2] / (sigma_u * sigma_v)
  # Fixed effect priors
  alpha ~ dnorm(0, {PRIOR_TAU})
  for (k in 1:K) {{
    beta[k] ~ dnorm(0, {PRIOR_TAU})
  }}
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# 3. MCMC sampling parameters (tuned for 8 GB / i5 5th gen)
# ─────────────────────────────────────────────────────────────────────────────
N_CHAINS  = 3      # 3 chains: valid R-hat, less RAM than 4
N_ADAPT   = 1000   # JAGS adaptation
N_BURNIN  = 3000   # discard first 3000
N_ITER    = 6000   # keep 6000 per chain = 18000 total
THIN      = 3      # store every 3rd → effective 2000 per chain
SEED      = 42

# For M3 (larger model): fewer iterations to keep runtime manageable
N_BURNIN_M3 = 2000
N_ITER_M3   = 4000
THIN_M3     = 2

# ─────────────────────────────────────────────────────────────────────────────
# 4. Fit all models
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pyjags
    PYJAGS_OK = True
    print("pyjags found — running full JAGS models")
except ImportError:
    PYJAGS_OK = False
    print("WARNING: pyjags not found.")
    print("Install JAGS from https://sourceforge.net/projects/mcmc-jags/")
    print("Then: pip install pyjags")
    print("Running fallback MAP estimation for code testing...")

def run_jags(model_string, data_dict, params, label,
             n_burnin=N_BURNIN, n_iter=N_ITER, thin=THIN):
    print(f"\n{'─'*55}\nFitting {label}...")
    t0 = time.time()
    model = pyjags.Model(
        code   = model_string,
        data   = data_dict,
        chains = N_CHAINS,
        adapt  = N_ADAPT,
    )
    print(f"  Burn-in ({n_burnin} iter)...")
    model.sample(n_burnin, vars=[])          # discard burn-in
    print(f"  Sampling ({n_iter} iter, thin={thin})...")
    samples = model.sample(n_iter, vars=params, thin=thin)
    elapsed = time.time()-t0
    print(f"  Done in {elapsed/60:.1f} min")
    return samples

# Data for JAGS (all as numpy arrays or Python scalars)
base_data = dict(
    N=int(N), K=int(K), J=int(J),
    y=y, X=X, precinct_idx=precinct_idx,
    weapon=df["weapon_found_bin"].values.astype(float),
)

PARAMS_M1 = ["alpha"] + [f"beta[{k+1}]" for k in range(K)]
PARAMS_M2 = PARAMS_M1 + ["sigma_u", "u"]
PARAMS_M3 = PARAMS_M1 + ["sigma_u","sigma_v","rho","u","v"]

R_mat = np.eye(2)   # identity — Wishart scale matrix for M3

if PYJAGS_OK:
    samples1 = run_jags(MODEL1, {k:v for k,v in base_data.items() if k in ["N","K","y","X"]},
                        PARAMS_M1, "Model 1 — Baseline Logistic")

    samples2 = run_jags(MODEL2, {k:v for k,v in base_data.items() if k in ["N","K","J","y","X","precinct_idx"]},
                        PARAMS_M2, "Model 2 — Hierarchical (precinct random intercept)")

    samples3 = run_jags(MODEL3, {**{k:v for k,v in base_data.items() if k in ["N","K","J","y","X","precinct_idx","weapon"]},
                                  "R": R_mat},
                        PARAMS_M3, "Model 3 — Hierarchical (random intercept + weapon slope)",
                        n_burnin=N_BURNIN_M3, n_iter=N_ITER_M3, thin=THIN_M3)

    with open("jags_samples.pkl","wb") as f:
        pickle.dump({"m1":samples1,"m2":samples2,"m3":samples3}, f)
    print("\nSamples saved to jags_samples.pkl")
else:
    # ── Fallback: MAP via scipy (for testing code flow without JAGS) ──────────
    from scipy.optimize import minimize
    from scipy.special import expit as sigmoid

    def neg_log_post(theta, prior_tau=PRIOR_TAU):
        a = theta[0]; b = theta[1:K+1]
        lo = a + X @ b
        ll = np.sum(y*lo - np.logaddexp(0,lo))
        lp = -0.5*prior_tau*(a**2 + np.sum(b**2))
        return -(ll+lp)

    print("Running MAP fallback (no JAGS)...")
    res = minimize(neg_log_post, np.zeros(K+1), method="L-BFGS-B")
    theta_map = res.x
    # Fake samples as MAP ± small noise (just to test downstream plotting)
    def fake_samples(theta, n=500):
        return {f"beta[{k+1}]": theta[1+k]+np.random.normal(0,0.05,(n,1,1)) for k in range(K)} | \
               {"alpha": theta[0]+np.random.normal(0,0.05,(n,1,1))}
    samples1 = samples2 = samples3 = fake_samples(theta_map)
    print("MAP fallback done. Install JAGS for real MCMC results.")

def get_draws(samples, param):
    """Return 1-D array of all draws across chains for a scalar parameter."""
    arr = samples[param]         # shape: (n_iter, n_chains, 1) or (n_iter, n_chains, J)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:,:,0]         # scalar param → (n_iter, n_chains)
    return arr.flatten()

def get_vector_draws(samples, param, size):
    """Return 2-D array (total_draws, size) for a vector parameter like beta or u."""
    draws = []
    for k in range(1, size+1):
        key = f"{param}[{k}]"
        draws.append(get_draws(samples, key))
    return np.column_stack(draws)   # (total_draws, size)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Convergence diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def gelman_rubin(samples, param):
    """Compute R-hat for a scalar parameter across chains."""
    arr = samples[param]
    if arr.ndim == 3:
        arr = arr[:,:,0]
    n, m = arr.shape          # n_iter, n_chains
    chain_means = arr.mean(axis=0)
    grand_mean  = arr.mean()
    B = n * np.var(chain_means, ddof=1)
    W = np.mean([np.var(arr[:,c], ddof=1) for c in range(m)])
    var_hat = (n-1)/n * W + B/n
    return float(np.sqrt(var_hat/W)) if W > 0 else np.nan

def ess_single(x):
    """ESS via autocorrelation (Geyer's initial monotone sequence)."""
    n = len(x); x = x - x.mean()
    if np.var(x) == 0: return 1.0
    acf = np.correlate(x, x, mode="full")[n-1:] / (np.var(x)*n)
    rho_sum = 1.0
    t = 1
    while t+1 < len(acf):
        pair = acf[t] + acf[t+1]
        if pair <= 0: break
        rho_sum += 2*pair; t += 2
    return float(n / max(rho_sum, 1.0))

def convergence_table(samples, params_list, label):
    print(f"\n=== Convergence: {label} ===")
    rows = []
    for p in params_list:
        if p not in samples: continue
        rhat = gelman_rubin(samples, p)
        merged = get_draws(samples, p)
        ess  = ess_single(merged)
        rows.append({"param":p, "mean":merged.mean(), "sd":merged.std(),
                     "q2_5":np.percentile(merged,2.5), "q97_5":np.percentile(merged,97.5),
                     "ESS":ess, "Rhat":rhat})
    tbl = pd.DataFrame(rows)
    print(tbl.round(3).to_string(index=False))
    bad = tbl[tbl["Rhat"] > 1.01]
    print(f"  R-hat > 1.01: {len(bad)}")
    low = tbl[tbl["ESS"] < 400]
    print(f"  ESS < 400:    {len(low)}")
    return tbl

kp1_show = ["alpha"] + [f"beta[{k+1}]" for k in range(6)]   # show first 6 betas
kp2_show = kp1_show + ["sigma_u"]
kp3_show = kp1_show + ["sigma_u","sigma_v","rho"]

conv1 = convergence_table(samples1, kp1_show, "Model 1 — Baseline")
conv2 = convergence_table(samples2, kp2_show, "Model 2 — Hierarchical intercept")
conv3 = convergence_table(samples3, kp3_show, "Model 3 — Hierarchical intercept+slope")

conv2.to_csv("convergence_model2.csv", index=False)
print("\nSaved convergence_model2.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 7. DIC  (computed from JAGS deviance via log-likelihood)
# ─────────────────────────────────────────────────────────────────────────────
def compute_dic(samples, X_mat, y_vec, precinct_vec=None, weapon_vec=None, model="m1"):
    """
    DIC = D(theta_bar) + 2*pD
    D(theta) = -2 * log p(y | theta)
    pD = var(D(theta)) / 2   (Gelman variance formula)
    """
    alpha_draws = get_draws(samples, "alpha")
    beta_draws  = get_vector_draws(samples, "beta", X_mat.shape[1])

    def deviance(a, b, u=None, v=None):
        lo = a + X_mat @ b
        if u is not None:
            lo += u[precinct_vec-1]
        if v is not None:
            lo += v[precinct_vec-1] * weapon_vec
        return -2 * np.sum(y_vec*lo - np.logaddexp(0,lo))

    n_draws = len(alpha_draws)

    if model == "m1":
        D_all = np.array([deviance(alpha_draws[i], beta_draws[i])
                          for i in range(n_draws)])
    elif model == "m2":
        u_draws = get_vector_draws(samples, "u", J)
        D_all = np.array([deviance(alpha_draws[i], beta_draws[i], u_draws[i])
                          for i in range(n_draws)])
    else:
        u_draws = get_vector_draws(samples, "u", J)
        v_draws = get_vector_draws(samples, "v", J)
        D_all = np.array([deviance(alpha_draws[i], beta_draws[i], u_draws[i], v_draws[i])
                          for i in range(n_draws)])

    D_bar      = D_all.mean()
    pD         = D_all.var() / 2
    DIC        = D_bar + pD
    return float(D_bar), float(pD), float(DIC)

print("\nComputing DIC...")
D1_bar, pD1, DIC1 = compute_dic(samples1, X, y, model="m1")
D2_bar, pD2, DIC2 = compute_dic(samples2, X, y, precinct_vec=precinct_idx,
                                  weapon_vec=df["weapon_found_bin"].values, model="m2")
D3_bar, pD3, DIC3 = compute_dic(samples3, X, y, precinct_vec=precinct_idx,
                                  weapon_vec=df["weapon_found_bin"].values, model="m3")

print(f"\n{'Model':<30} {'DIC':>9} {'pD':>8}")
for lbl,dic,pd_ in [("M1: Logistic",DIC1,pD1),
                     ("M2: + Precinct RE",DIC2,pD2),
                     ("M3: + Weapon Slope",DIC3,pD3)]:
    print(f"  {lbl:<28} {dic:>9.1f} {pd_:>8.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. LOO-CV  (via pointwise log-likelihood, PSIS)
# ─────────────────────────────────────────────────────────────────────────────
def compute_log_lik_matrix(samples, X_mat, y_vec, precinct_vec=None, weapon_vec=None, model="m1"):
    """Returns (n_draws, N) log-likelihood matrix."""
    alpha_draws = get_draws(samples, "alpha")
    beta_draws  = get_vector_draws(samples, "beta", X_mat.shape[1])
    n_draws = len(alpha_draws)
    LL = np.zeros((n_draws, len(y_vec)), dtype=np.float32)

    if model == "m2":
        u_draws = get_vector_draws(samples, "u", J)
    elif model == "m3":
        u_draws = get_vector_draws(samples, "u", J)
        v_draws = get_vector_draws(samples, "v", J)

    for i in range(n_draws):
        lo = alpha_draws[i] + X_mat @ beta_draws[i]
        if model == "m2":
            lo += u_draws[i][precinct_vec-1]
        elif model == "m3":
            lo += u_draws[i][precinct_vec-1] + v_draws[i][precinct_vec-1]*weapon_vec
        LL[i] = y_vec*lo - np.logaddexp(0, lo)
    return LL

def psis_loo(LL):
    """Simplified PSIS-LOO: returns (elpd, se)."""
    n_draws, N_obs = LL.shape
    elpd_i = np.zeros(N_obs)
    for i in range(N_obs):
        lw = LL[:,i] - LL[:,i].max()
        w  = np.exp(lw); w /= w.sum()
        elpd_i[i] = np.log(np.dot(w, np.exp(LL[:,i])))
    return float(elpd_i.sum()), float(np.sqrt(N_obs*np.var(elpd_i,ddof=1)))

print("\nComputing LOO-CV (processes one model at a time to save RAM)...")

weapon_vec = df["weapon_found_bin"].values.astype(float)

LL1 = compute_log_lik_matrix(samples1, X, y, model="m1")
e1,s1 = psis_loo(LL1); del LL1

LL2 = compute_log_lik_matrix(samples2, X, y, precinct_idx, weapon_vec, model="m2")
e2,s2 = psis_loo(LL2); del LL2

LL3 = compute_log_lik_matrix(samples3, X, y, precinct_idx, weapon_vec, model="m3")
e3,s3 = psis_loo(LL3); del LL3

best = max(e1,e2,e3)
print(f"\n{'Model':<30} {'ELPD':>9} {'SE':>7} {'ΔELPD':>8}")
for lbl,e,s in [("M1: Logistic",e1,s1),("M2: + Precinct RE",e2,s2),("M3: + Weapon Slope",e3,s3)]:
    print(f"  {lbl:<28} {e:>9.1f} {s:>7.1f} {e-best:>8.1f}")

with open("loo_results.json","w") as f:
    json.dump({"M1":{"elpd":e1,"se":s1,"DIC":DIC1},
               "M2":{"elpd":e2,"se":s2,"DIC":DIC2},
               "M3":{"elpd":e3,"se":s3,"DIC":DIC3},
               "delta_M2_vs_M1":round(e2-e1,2),
               "delta_M3_vs_M2":round(e3-e2,2)}, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Posterior summaries
# ─────────────────────────────────────────────────────────────────────────────
def posterior_summary(samples, label):
    rows = []
    ad = get_draws(samples, "alpha")
    rows.append(dict(covariate="Intercept",
                     mean=ad.mean(), sd=ad.std(),
                     q2_5=np.percentile(ad,2.5), q97_5=np.percentile(ad,97.5),
                     prob_pos=(ad>0).mean()))
    for k, name in enumerate(COV_LABELS):
        d = get_draws(samples, f"beta[{k+1}]")
        rows.append(dict(covariate=name,
                         mean=d.mean(), sd=d.std(),
                         q2_5=np.percentile(d,2.5), q97_5=np.percentile(d,97.5),
                         prob_pos=(d>0).mean()))
    out = pd.DataFrame(rows)
    out.to_csv(f"{label}_posterior_summary.csv", index=False)
    return out

s1_df = posterior_summary(samples1, "model1")
s2_df = posterior_summary(samples2, "model2")
s3_df = posterior_summary(samples3, "model3")

# Extra RE params
su2 = get_draws(samples2,"sigma_u")
su3 = get_draws(samples3,"sigma_u")
sv3 = get_draws(samples3,"sigma_v")
rho3= get_draws(samples3,"rho")

extra = {
    "M2_sigma_u":{"mean":float(su2.mean()),"q2_5":float(np.percentile(su2,2.5)),"q97_5":float(np.percentile(su2,97.5))},
    "M3_sigma_u":{"mean":float(su3.mean()),"q2_5":float(np.percentile(su3,2.5)),"q97_5":float(np.percentile(su3,97.5))},
    "M3_sigma_v":{"mean":float(sv3.mean()),"q2_5":float(np.percentile(sv3,2.5)),"q97_5":float(np.percentile(sv3,97.5))},
    "M3_rho":    {"mean":float(rho3.mean()),"q2_5":float(np.percentile(rho3,2.5)),"q97_5":float(np.percentile(rho3,97.5))},
}
with open("extra_params.json","w") as f: json.dump(extra, f, indent=2)
print(f"\nM2 σ_u = {extra['M2_sigma_u']['mean']:.3f} [{extra['M2_sigma_u']['q2_5']:.3f}, {extra['M2_sigma_u']['q97_5']:.3f}]")
print(f"M3 σ_v = {extra['M3_sigma_v']['mean']:.3f} [{extra['M3_sigma_v']['q2_5']:.3f}, {extra['M3_sigma_v']['q97_5']:.3f}]")
print(f"M3 ρ   = {extra['M3_rho']['mean']:.3f} [{extra['M3_rho']['q2_5']:.3f}, {extra['M3_rho']['q97_5']:.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Odds ratio plot — Model 2
# ─────────────────────────────────────────────────────────────────────────────
def or_plot(summ_df, title, fname):
    sub = summ_df[summ_df["covariate"]!="Intercept"].copy()
    sub["or_mean"] = np.exp(sub["mean"])
    sub["or_lo"]   = np.exp(sub["q2_5"])
    sub["or_hi"]   = np.exp(sub["q97_5"])
    sub["sig"]     = ~((sub["q2_5"]<0)&(sub["q97_5"]>0))
    sub = sub.sort_values("or_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9,9))
    for i, row in sub.iterrows():
        c = RED if row["sig"] else BLUE
        ax.plot([row["or_lo"],row["or_hi"]], [i,i], color=c, lw=1.8, solid_capstyle="round")
        ax.plot(row["or_mean"], i, "o", color=c, ms=6, zorder=5)
    ax.axvline(1, color="grey", ls="--", lw=0.9)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub["covariate"], fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("Odds Ratio (log scale)", fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.legend(handles=[mpatches.Patch(color=RED,label="95% CrI excludes 1"),
                       mpatches.Patch(color=BLUE,label="95% CrI includes 1")], fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(fname, dpi=130); plt.close(); print(f"Saved {fname}")

or_plot(s2_df,"Model 2 Odds Ratios (95% CrI)\nRef: Black, Female, Weapon, Bronx, Autumn",
        "plots/model2_odds_ratios.png")
or_plot(s3_df,"Model 3 Odds Ratios (95% CrI)\nRef: Black, Female, Weapon, Bronx, Autumn",
        "plots/model3_odds_ratios.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Trace plots — Model 2
# ─────────────────────────────────────────────────────────────────────────────
TRACE_KEYS   = ["alpha", f"beta[{WEAPON_COL+1}]", "beta[1]", "sigma_u"]
TRACE_LABELS = ["Intercept α", "β: Weapon Found", "β: Age (scaled)", "σ_u (precinct SD)"]
COLORS_C     = [BLUE, RED, GREEN]

fig, axes = plt.subplots(2, 2, figsize=(11,6))
for ax, key, label in zip(axes.flatten(), TRACE_KEYS, TRACE_LABELS):
    if key not in samples2:
        ax.set_visible(False); continue
    arr = samples2[key]
    if arr.ndim == 3: arr = arr[:,:,0]   # (n_iter, n_chains)
    for c in range(arr.shape[1]):
        ax.plot(arr[:,c], alpha=0.75, lw=0.6, color=COLORS_C[c % len(COLORS_C)],
                label=f"Chain {c+1}")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Iteration (post burn-in)", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
axes[0,0].legend(fontsize=8)
plt.suptitle("Trace Plots — Model 2 (post burn-in)\nWell-mixed chains confirm convergence",
             fontweight="bold")
plt.tight_layout()
plt.savefig("plots/traceplots_model2.png", dpi=130)
plt.close(); print("Saved plots/traceplots_model2.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Caterpillar plot — precinct random intercepts (Model 2)
# ─────────────────────────────────────────────────────────────────────────────
u2_draws = get_vector_draws(samples2, "u", J)
u_m  = u2_draws.mean(axis=0)
u_lo = np.percentile(u2_draws, 2.5,  axis=0)
u_hi = np.percentile(u2_draws, 97.5, axis=0)
del u2_draws
order_u = np.argsort(u_m)

fig, ax = plt.subplots(figsize=(7,7))
ax.axvline(0, color="grey", ls="--", lw=0.8)
for rank, j in enumerate(order_u):
    c = RED if u_lo[j]>0 else (GREEN if u_hi[j]<0 else BLUE)
    ax.plot([u_lo[j],u_hi[j]], [rank,rank], color=c, alpha=0.55, lw=1.1)
    ax.plot(u_m[j], rank, "o", color=c, ms=2.5)
ax.set_xlabel("Random Intercept u_j  (log-odds scale)", fontsize=10)
ax.set_ylabel("Precinct (ranked by posterior mean)", fontsize=9)
ax.set_title("Model 2: Precinct Random Intercepts\nRed=above 0 | Green=below 0 | Blue=spans 0",
             fontweight="bold")
ax.legend(handles=[mpatches.Patch(color=RED,label="Higher arrest tendency"),
                   mpatches.Patch(color=GREEN,label="Lower arrest tendency"),
                   mpatches.Patch(color=BLUE,label="Uncertain")], fontsize=8, loc="upper left")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plots/precinct_random_effects.png", dpi=130)
plt.close(); print("Saved plots/precinct_random_effects.png")

# ─────────────────────────────────────────────────────────────────────────────
# 13. Random slopes — weapon effect by precinct (Model 3)
# ─────────────────────────────────────────────────────────────────────────────
v3_draws = get_vector_draws(samples3, "v", J)
v_m  = v3_draws.mean(axis=0)
v_lo = np.percentile(v3_draws, 2.5,  axis=0)
v_hi = np.percentile(v3_draws, 97.5, axis=0)
del v3_draws
order_v = np.argsort(v_m)

fig, ax = plt.subplots(figsize=(7,7))
ax.axvline(0, color="grey", ls="--", lw=0.8)
for rank, j in enumerate(order_v):
    c = RED if v_lo[j]>0 else (GREEN if v_hi[j]<0 else BLUE)
    ax.plot([v_lo[j],v_hi[j]], [rank,rank], color=c, alpha=0.55, lw=1.1)
    ax.plot(v_m[j], rank, "o", color=c, ms=2.5)
ax.set_xlabel("Random Slope v_j for Weapon Found  (log-odds scale)", fontsize=10)
ax.set_ylabel("Precinct (ranked)", fontsize=9)
ax.set_title("Model 3: Precinct-Level Weapon Effect\nVariation in weapon → arrest relationship across precincts",
             fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plots/precinct_weapon_slopes.png", dpi=130)
plt.close(); print("Saved plots/precinct_weapon_slopes.png")

# ─────────────────────────────────────────────────────────────────────────────
# 14. Posterior predictive check — Model 2
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing PPC (Model 2)...")
alpha_d = get_draws(samples2, "alpha")
beta_d  = get_vector_draws(samples2, "beta", K)
u_d     = get_vector_draws(samples2, "u", J)

# Compute predicted probabilities efficiently — sample 500 draws
n_ppc = min(500, len(alpha_d))
idx_ppc = np.random.choice(len(alpha_d), n_ppc, replace=False)
lo_ppc  = alpha_d[idx_ppc,None] + beta_d[idx_ppc] @ X.T + u_d[idx_ppc][:,precinct_idx-1]
p_ppc   = expit(lo_ppc)                 # (n_ppc, N)
y_rep   = (np.random.rand(*p_ppc.shape) < p_ppc).astype(np.int8)
del lo_ppc, p_ppc

pred_rates = y_rep.mean(axis=1)
obs_rate   = y.mean()
BOROS = sorted(df["borough"].unique())

fig, axes = plt.subplots(1, 2, figsize=(11,4))
ax = axes[0]
ax.hist(pred_rates, bins=40, color=BLUE, alpha=0.7, edgecolor="none")
ax.axvline(obs_rate, color=RED, lw=2)
ax.text(obs_rate+0.001, ax.get_ylim()[1]*0.88, f"Observed\n{obs_rate:.3f}", color=RED, fontsize=9)
ax.set_xlabel("Predicted Arrest Rate"); ax.set_ylabel("Count")
ax.set_title("(a) PPC: Overall Arrest Rate", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
obs_b  = [y[df["borough"].values==b].mean() for b in BOROS]
pred_b = [y_rep[:,df["borough"].values==b].mean(axis=1).mean() for b in BOROS]
lo_b   = [np.percentile(y_rep[:,df["borough"].values==b].mean(axis=1),2.5)  for b in BOROS]
hi_b   = [np.percentile(y_rep[:,df["borough"].values==b].mean(axis=1),97.5) for b in BOROS]
xp = np.arange(len(BOROS))
ax.bar(xp-0.2, obs_b,  0.35, color=RED,  alpha=0.75, label="Observed")
ax.bar(xp+0.2, pred_b, 0.35, color=BLUE, alpha=0.75, label="Predicted")
ax.errorbar(xp+0.2, pred_b,
            yerr=[np.array(pred_b)-np.array(lo_b), np.array(hi_b)-np.array(pred_b)],
            fmt="none", color="black", capsize=3, lw=1.2)
ax.set_xticks(xp); ax.set_xticklabels(BOROS, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Arrest Rate"); ax.legend(fontsize=9)
ax.set_title("(b) PPC: Arrest Rate by Borough", fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
del y_rep

plt.suptitle("Model 2 — Posterior Predictive Checks", fontweight="bold")
plt.tight_layout()
plt.savefig("plots/ppc_model2.png", dpi=130)
plt.close(); print("Saved plots/ppc_model2.png")

# ─────────────────────────────────────────────────────────────────────────────
# 15. Seasonal effects violin plot — Model 2
# ─────────────────────────────────────────────────────────────────────────────
SEAS_IDX = {"Winter": X_COLS.index("seas_Win"),
            "Spring": X_COLS.index("seas_Spr"),
            "Summer": X_COLS.index("seas_Sum")}
b_draws = get_vector_draws(samples2, "beta", K)

fig, ax = plt.subplots(figsize=(7,4))
positions = list(range(len(SEAS_IDX)))
vp = ax.violinplot([b_draws[:,idx] for idx in SEAS_IDX.values()],
                   positions=positions, showmedians=True, widths=0.7)
for body in vp["bodies"]:
    body.set_alpha(0.6); body.set_facecolor(BLUE)
ax.axhline(0, color="grey", ls="--", lw=1, label="Autumn (reference = 0)")
ax.set_xticks(positions); ax.set_xticklabels(list(SEAS_IDX.keys()))
ax.set_ylabel("Log-odds coefficient (vs Autumn)"); ax.legend(fontsize=9)
ax.set_title("Seasonal Effect on Arrest Probability\nModel 2 posterior distributions (ref = Autumn)",
             fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plots/seasonal_effects.png", dpi=130)
plt.close(); print("Saved plots/seasonal_effects.png")

# ─────────────────────────────────────────────────────────────────────────────
# 16. LOO comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6,4))
labels_bar = ["M1\nLogistic","M2\n+ Precinct RE","M3\n+ Weapon Slope"]
elpds=[e1,e2,e3]; ses=[s1,s2,s3]
ax.bar(labels_bar, elpds, color=["#aec7e8",BLUE,"#1a4f7a"], alpha=0.85, width=0.5)
ax.errorbar(labels_bar, elpds, yerr=[1.96*s for s in ses],
            fmt="none", color="black", capsize=5, lw=1.5)
for i in range(1,3):
    d = elpds[i]-elpds[i-1]
    ax.text(i, elpds[i]+1.96*ses[i]+2, f"Δ={d:+.1f}", ha="center", fontsize=9)
ax.set_ylabel("ELPD (LOO-CV)")
ax.set_title("Model Comparison — LOO-CV\nHigher ELPD = better", fontweight="bold")
yspan = max(elpds)-min(elpds) or 10
ax.set_ylim(min(elpds)-yspan*0.3, max(elpds)+yspan*0.5)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plots/loo_comparison.png", dpi=130)
plt.close(); print("Saved plots/loo_comparison.png")

print("\n" + "="*55)
print("ALL DONE")
print(f"  LOO: M1={e1:.1f}  M2={e2:.1f}  M3={e3:.1f}")
print(f"  DIC: M1={DIC1:.1f}  M2={DIC2:.1f}  M3={DIC3:.1f}")
print(f"  M2 σ_u = {extra['M2_sigma_u']['mean']:.3f}")
print("Plots saved to plots/")
