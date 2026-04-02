import numpy as np
from scipy.stats import norm


def gen_dif_mean(N, time, meanf1, meanf2, sd):
    """
    Scenario 3 (Setting 5): two groups with different mean curve functions.
    Direct translation of gendifmean() from Xie & Ogden (2024).

    R original:
        discrete_data1 = meanf1(time) + matrix(rnorm(N/2*J, sd=sd), J, N/2)
        m1 = mean(discrete_data1)          # grand mean (scalar)
        discrete_data1 = discrete_data1 - m1
        ... (same for group 2)
        discrete_data = cbind(data1, data2)  # (J, N)
        class = c(rep(1, N/2), rep(-1, N/2))

    Parameters
    ----------
    N : int
        Total sample size — must be even; N/2 per group.
    time : ndarray (J,)
        Evenly spaced time g                                                                                                                                                   
import matplotlib.pyplot as plt                                                                                                                                                                 
import matplotlib.patches as mpatches                                                                                                                                                           
from sklearn.svm import SVC                                                                                                                                                                     
from genData import gen_dif_mean                                                                                                                                                              
from fsvm_implement import fsvc, evaluate_blind_test                                                                                                                                            
                                                                                                                                                                                                
# ── Mean functions (paper p.1185) ─────────────────────────────────────────                                                                                                                    
meanf1 = lambda t: np.cos(10*t - np.pi/4) + 0.5*np.sin(8*t  - np.pi/4)                                                                                                                          
meanf2 = lambda t: np.cos(8*t  - np.pi/4) + 0.5*np.sin(10*t - np.pi/4)                                                                                                                          
                                                                                                                                                                                                
# ── Simulation grid ────────────────────────────────────────────────────────                                                                                                                   
settings   = [(50, 40), (50, 100), (100, 50)]   # (N, J)                                                                                                                                        
noise_sds  = [1.0, np.sqrt(10)]                  # N(0,1) and N(0,10) variance                                                                                                                  
N_REPS     = 20    # paper uses 100; start with 20 to verify trend                                                                                                                              
N_TEST     = 1000  # paper uses 10000; 1000 sufficient to verify ordering                                                                                                                       
                                                                                                                                                                                                
# FSVC hyperparameter grids (paper values)                                                                                                                                                      
SMOOTHERS = [0.5, 1.0, 5.0, 10.0]                                                                                                                                                               
CS        = [0.01, 0.2575, 0.505, 0.7525, 1.0]                                                                                                                                                  
KS        = list(range(1, 6))                                                                                                                                                                   
                                                                                                                                                                                                
METHODS = ["SVC\nlinear", "SVC\nGaussian", "FSVC\nlinear", "FSVC\nGaussian"]                                                                                                                    
COLORS  = ["#E41A1C", "#FF7F00", "#4DAF4A", "#377EB8"]                                                                                                                                          
                                                                                                                                                                                                
# ── Run simulations ────────────────────────────────────────────────────────                                                                                                                 
# results[noise_idx][setting_idx][method_idx] = list of accuracies                                                                                                                              
results = [[[ [] for _ in METHODS] for _ in settings] for _ in noise_sds]                                                                                                                       

total = N_REPS * len(settings) * len(noise_sds)                                                                                                                                                 
done  = 0                                                                                                                                                                                     
                                                                                                                                                                                                
for n_idx, sd in enumerate(noise_sds):                                                                                                                                                          
    for s_idx, (N, J) in enumerate(settings):
        time = np.linspace(0, 1, J)                                                                                                                                                             
                                                                                                                                                                                                
        for rep in range(N_REPS):
            # Generate data                                                                                                                                                                     
            train = gen_dif_mean(N, time, meanf1, meanf2, sd)                                                                                                                                 
            test  = gen_dif_mean(N_TEST, time, meanf1, meanf2, sd)                                                                                                                              

            X_train = train["discrete_data"].T   # (N, J)                                                                                                                                       
            y_train = train["classlabel"]                                                                                                                                                     
            X_test  = test["discrete_data"].T                                                                                                                                                   
            y_test  = test["classlabel"]                                                                                                                                                        

            # ── SVC linear ────────────────────────────────────────────                                                                                                                        
            svc_lin = SVC(kernel="linear")                                                                                                                                                    
            svc_lin.fit(X_train, y_train)
            results[n_idx][s_idx][0].append(                                                                                                                                                    
                np.mean(svc_lin.predict(X_test) == y_test))
                                                                                                                                                                                                
            # ── SVC Gaussian ──────────────────────────────────────────                                                                                                                      
            svc_rbf = SVC(kernel="rbf")                                                                                                                                                         
            svc_rbf.fit(X_train, y_train)                                                                                                                                                     
            results[n_idx][s_idx][1].append(                                                                                                                                                    
                np.mean(svc_rbf.predict(X_test) == y_test))
                                                                                                                                                                                                
            # ── FSVC linear ───────────────────────────────────────────                                                                                                                        
            m_lin = fsvc(X_train, y_train, use_r=True, kernel="linear",
                        smoothers=SMOOTHERS, Cs=CS, Ks=KS,                                                                                                                                     
                        npc=5, n_folds=5, random_state=rep)                                                                                                                                  
            results[n_idx][s_idx][2].append(                                                                                                                                                    
                evaluate_blind_test(m_lin, X_test, y_test)["accuracy"])                                                                                                                       
                                                                                                                                                                                                
            # ── FSVC Gaussian ─────────────────────────────────────────                                                                                                                      
            m_rbf = fsvc(X_train, y_train, use_r=True, kernel="rbf",                                                                                                                            
                        smoothers=SMOOTHERS, Cs=CS, Ks=KS,                                                                                                                                     
                        npc=5, n_folds=5, random_state=rep)
            results[n_idx][s_idx][3].append(                                                                                                                                                    
                evaluate_blind_test(m_rbf, X_test, y_test)["accuracy"])                                                                                                                         

            done += 1                                                                                                                                                                           
            print(f"  [{done}/{total}]  sd={sd:.2f}  N={N} J={J}  rep={rep+1}")                                                                                                               
                                                                                                                                                                                                
# ── Plot — replicating Figure 2(a) layout ─────────────────────────────────                                                                                                                    
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)                                                                                                                                    
noise_labels = [r"$\eta_{ij} \sim N(0,1)$", r"$\eta_{ij} \sim N(0,10)$"]                                                                                                                        
x_labels     = [f"N={N},J={J}" for N, J in settings]                                                                                                                                            
                                                                                                                                                                                                
W       = 0.15   # box width                                                                                                                                                                    
OFFSETS = [-1.5*W, -0.5*W, 0.5*W, 1.5*W]   # 4 methods side by side per group                                                                                                                   
                                                                                                                                                                                                
for ax, n_idx, noise_lbl in zip(axes, range(2), noise_labels):                                                                                                                                
    for s_idx in range(len(settings)):                                                                                                                                                          
        x_center = s_idx + 1                                                                                                                                                                    
        for m_idx in range(len(METHODS)):
            data = results[n_idx][s_idx][m_idx]                                                                                                                                                 
            bp = ax.boxplot(                                                                                                                                                                  
                data,                                                                                                                                                                           
                positions=[x_center + OFFSETS[m_idx]],                                                                                                                                        
                widths=W,                                                                                                                                                                       
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),                                                                                                                                 
                whiskerprops=dict(linewidth=1),                                                                                                                                                 
                capprops=dict(linewidth=1),
                flierprops=dict(marker="o", markersize=3, alpha=0.5),                                                                                                                           
            )                                                                                                                                                                                   
            bp["boxes"][0].set_facecolor(COLORS[m_idx])
            bp["boxes"][0].set_alpha(0.8)                                                                                                                                                       
                                                                                                                                                                                                
    ax.set_xticks(range(1, len(settings) + 1))
    ax.set_xticklabels(x_labels, fontsize=10)                                                                                                                                                   
    ax.set_title(noise_lbl, fontsize=11)                                                                                                                                                        
    ax.set_ylabel("Accuracy" if n_idx == 0 else "")
    ax.set_ylim(0.4, 1.05)                                                                                                                                                                      
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)                                                                                                                     
    ax.grid(axis="y", alpha=0.3)                                                                                                                                                                
                                                                                                                                                                                                
# Legend                                                                                                                                                                                        
patches = [mpatches.Patch(color=c, label=m.replace("\n", " "))                                                                                                                                  
            for c, m in zip(COLORS, METHODS)]                                                                                                                                                    
fig.legend(handles=patches, loc="upper center", ncol=4,
            fontsize=9, bbox_to_anchor=(0.5, 1.02))                                                                                                                                              
                                                                                                                                                                                                
fig.suptitle("Scenario 3 (Setting 5): Different Mean Functions", y=1.06, fontsize=12)                                                                                                           
plt.tight_layout()                                                                                                                                                                              
plt.savefig("fig2a_scenario3_replication.png", dpi=150, bbox_inches="tight")                                                                                                                    
plt.show()                                                                                                                                                                                      

# ── Print median summary ───────────────────────────────────────────────────                                                                                                                   
print("\nMedian accuracy summary:")                                                                                                                                                           
for n_idx, sd in enumerate(noise_sds):                                                                                                                                                          
    print(f"\n  noise sd={sd:.2f}")
    for s_idx, (N, J) in enumerate(settings):                                                                                                                                                   
        print(f"    N={N} J={J}: ", end="")                                                                                                                                                   
        for m_idx, method in enumerate(METHODS):                                                                                                                                                
            med = np.median(results[n_idx][s_idx][m_idx])                                                                                                                                       
            print(f"{method.replace(chr(10),' ')}={med:.3f}  ", end="")
        print()                                                                                                                                                                                 
                                                                    rid on [0, 1].
    meanf1, meanf2 : callable
        Mean functions for group 1 and 2.  Each takes (J,) and returns (J,).
    sd : float
        Noise standard deviation (rnorm sd parameter).

    Returns
    -------
    dict with keys:
        'classlabel'    : ndarray (N,)   — {+1, -1}
        'discrete_data' : ndarray (J, N) — functional data, one column per subject
    """
    J    = len(time)
    half = N // 2

    # Group 1: mean + iid noise, then subtract grand mean (scalar)
    data1 = meanf1(time) + np.random.randn(half, J) * sd   # (half, J)
    data1 -= data1.mean()

    # Group 2
    data2 = meanf2(time) + np.random.randn(half, J) * sd
    data2 -= data2.mean()

    # Stack into (J, N) matching R's cbind(data1.T, data2.T)
    discrete_data = np.vstack([data1, data2]).T             # (J, N)

    classlabel = np.array([1] * half + [-1] * half)

    return {"classlabel": classlabel, "discrete_data": discrete_data}


def gen_fsvc_pca(n, k, bfun, lambdas, grids, eigenfunction, noise_sigma):
    """
    Generate simulated data for Functional SVC (Scenario 1, Settings 1-3 - XieOgden
    
    ).
    y is generated from a function of FPC scores.

    Parameters
    ----------
    n : int
        Sample size.
    k : int
        Number of FPCs.
    bfun : callable
        Boundary function taking two arrays (score1, score2) and returning an array.
    lambdas : array-like of shape (k,)
        Eigenvalues.
    grids : array-like
        Time grid points.
    eigenfunction : array-like of shape (k, n_time)
        Eigenfunctions evaluated on the time grid.
    noise_sigma : float
        Variance of noise in functional data.

    Returns
    -------
    dict with keys:
        'classlabel' : ndarray of shape (n,) — class labels {1, -1}
        'boundary'   : ndarray of shape (n,) — boundary function values
        'PCscore'    : ndarray of shape (n, k) — FPC scores
        'discrete_data' : ndarray of shape (n_time, n) — observed functional data
        'y'          : ndarray of shape (n,) — continuous response before thresholding
        'prob'       : ndarray of shape (n,) — probabilities
    """
    lambdas = np.asarray(lambdas, dtype=float)

    # Generate FPC scores: sqrt(lambda_i) * Z_i
    z = np.random.randn(n, k)  # (n, k)
    trainscore = z * np.sqrt(lambdas)  # (n, k), broadcasting

    # Compute discrete functional data: eigenfunction^T @ trainscore^T => (n_time, n)
    discrete_data = eigenfunction.T @ trainscore.T  # (n_time, n)

    # Add noise for each subject
    if noise_sigma > 0:
        noise = np.random.multivariate_normal(
            mean=np.zeros(len(grids)),
            cov=np.diag(np.full(len(grids), noise_sigma)),
            size=n,
        )  # (n, n_time)
        discrete_data = discrete_data + noise.T

    # Boundary function (uses first two FPC scores)
    boundary = bfun(trainscore[:, 0], trainscore[:, 1])

    # Add noise on y and compute class labels
    y = boundary + np.random.randn(n)
    prob = 1.0 / (1.0 + np.exp(-y))  # inverse logit
    classlabel = np.where(prob > 0.5, 1, -1)

    return {
        "classlabel": classlabel,
        "boundary": boundary,
        "PCscore": trainscore,
        "discrete_data": discrete_data,
        "y": y,
        "prob": prob,
    }