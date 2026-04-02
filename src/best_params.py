from sklearn.model_selection import LeaveOneOut, StratifiedKFold                                                                                                                                
from sklearn.decomposition import PCA                                                                                                                                                           
from sklearn.svm import SVC                                                                                                                                                                     
from sklearn.metrics import balanced_accuracy_score                                                                                                                                             
from itertools import product                                                                                                                                                                   
import numpy as np                                                                                                                                                                              

def fsvm_nested_cv(X, y, param_grid=None, inner_splits=5, random_state=42):                                                                                                                     
    """         
    Nested CV for FSVM:                                                                                                                                                                         
    - Outer loop: LOOCV → unbiased performance estimate
    - Inner loop: StratifiedKFold CV → grid search for best params per fold                                                                                                                     
    """                                                                                                                                                                                         
    if param_grid is None:
        param_grid = {                                                                                                                                                                          
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],                                                                                                                                                        
            'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]                                                                                                                      
        }                                                                                                                                                                                       
                                                                                                                                                                                                
    outer_loo = LeaveOneOut()                                                                                                                                                                   
    y_true, y_pred, decisions, best_params_per_fold = [], [], [], []
                                                                                                                                                                                                
    for fold, (train_idx, test_idx) in enumerate(outer_loo.split(X)):                                                                                                                           
        X_tr, X_te = X[train_idx], X[test_idx]                                                                                                                                                  
        y_tr, y_te = y[train_idx], y[test_idx]                                                                                                                                                  
                                                                                                                                                                                                
        # ---- Inner loop: grid search on training split ----                                                                                                                                   
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)                                                                                              
        best_inner_score = -1                                                                                                                                                                   
        best_params = {}                                                                                                                                                                        
                                                                                                                                                                                                
        for C, kernel, cw in product(param_grid['C'], param_grid['kernel'], param_grid['class_weight']):                                                                                        
            inner_preds, inner_true = [], []                                                                                                                                                    
                                                                                                                                                                                                
            for inner_train, inner_val in inner_cv.split(X_tr, y_tr):                                                                                                                           
                X_itr, X_ival = X_tr[inner_train], X_tr[inner_val]                                                                                                                              
                y_itr, y_ival = y_tr[inner_train], y_tr[inner_val]                                                                                                                              
                                                                                                                                                                                                
                mean_itr = X_itr.mean(axis=0)                                                                                                                                                   
                X_itr_c = X_itr - mean_itr                                                                                                                                                      
                X_ival_c = X_ival - mean_itr                                                                                                                                                    

                pca = PCA()                                                                                                                                                                     
                s_itr = pca.fit_transform(X_itr_c)
                K = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1                                                                                                         
                s_itr = s_itr[:, :K]                                                                                                                                                            
                s_ival = pca.transform(X_ival_c)[:, :K]                                                                                                                                         
                                                                                                                                                                                                
                clf = SVC(kernel=kernel, C=C, class_weight=cw)                                                                                                                                  
                clf.fit(s_itr, y_itr)
                inner_preds.extend((clf.decision_function(s_ival) > 0).astype(int))                                                                                                             
                inner_true.extend(y_ival)                                                                                                                                                       

            score = balanced_accuracy_score(inner_true, inner_preds)                                                                                                                            
            if score > best_inner_score:
                best_inner_score = score
                best_params = {'C': C, 'kernel': kernel, 'class_weight': cw}                                                                                                                    
                                                                                                                                                                                                
        # ---- Outer: train on full training split with best params ----                                                                                                                        
        mean_tr = X_tr.mean(axis=0)                                                                                                                                                             
        X_tr_c = X_tr - mean_tr                                                                                                                                                                 
        X_te_c = X_te - mean_tr
                                                                                                                                                                                                
        pca = PCA()                                                                                                                                                                             
        s_tr = pca.fit_transform(X_tr_c)                                                                                                                                                        
        K = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1                                                                                                                 
        s_tr = s_tr[:, :K]                                                                                                                                                                      
        s_te = pca.transform(X_te_c)[:, :K]                                                                                                                                                     
                                                                                                                                                                                                
        clf = SVC(**best_params)                                                                                                                                                                
        clf.fit(s_tr, y_tr)
        decision = clf.decision_function(s_te)[0]                                                                                                                                               
                                                                                                                                                                                                
        y_pred.append((decision > 0).astype(int))                                                                                                                                               
        y_true.append(y_te[0])                                                                                                                                                                  
        decisions.append(decision)                                                                                                                                                              
        best_params_per_fold.append(best_params)
                                                                                                                                                                                                
        print(f"Fold {fold+1}: best_params={best_params}, inner_BA={best_inner_score:.4f}")                                                                                                     

    results = {                                                                                                                                                                                 
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),                                                                                                                                                             
        "decisions": np.array(decisions),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),                                                                                                                           
        "best_params_per_fold": best_params_per_fold                                                                                                                                            
    }
                                                                                                                                                                                                
    print(f"\nOuter LOOCV Balanced Accuracy: {results['balanced_accuracy']:.4f}")                                                                                                               
    return results
                                                                                                                                                                                                     
# results = fsvm_nested_cv(X, y)
                            
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from itertools import product
import numpy as np

def fsvm_nested_cv_XieOgden(X, y, param_grid=None, inner_splits=5, random_state=42):
    """
    Nested CV for FSVM — fully joint tuning of K, C, kernel, class_weight.
    Follows Xie & Ogden (2024) Algorithm 1: K is tuned jointly with SVM
    hyperparameters rather than fixed by a variance threshold.

    - Outer loop: LOOCV  → unbiased performance estimate
    - Inner loop: StratifiedKFold CV → joint grid search over (K, C, kernel, class_weight)
    """
    if param_grid is None:
        param_grid = {
            'K':            list(range(1, 11)),          # ← CHANGED: K now in the grid (1 to 10, matching Xie & Ogden)
            'C':            [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel':       ['linear', 'rbf'],
            'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
        }

    outer_loo = LeaveOneOut()
    y_true, y_pred, decisions, best_params_per_fold = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(outer_loo.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # ---- Inner loop: joint grid search over (K, C, kernel, class_weight) ----
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        best_inner_score = -1
        best_params = {}

        # ← CHANGED: K is now unpacked from the grid alongside C, kernel, cw
        for K, C, kernel, cw in product(
            param_grid['K'], param_grid['C'],
            param_grid['kernel'], param_grid['class_weight']
        ):
            inner_preds, inner_true = [], []

            for inner_train, inner_val in inner_cv.split(X_tr, y_tr):
                X_itr, X_ival = X_tr[inner_train], X_tr[inner_val]
                y_itr, y_ival = y_tr[inner_train], y_tr[inner_val]

                mean_itr = X_itr.mean(axis=0)
                X_itr_c  = X_itr - mean_itr
                X_ival_c = X_ival - mean_itr

                pca = PCA()
                s_itr = pca.fit_transform(X_itr_c)

                # ← CHANGED: use the grid value of K directly; clip to available components
                K_safe = min(K, s_itr.shape[1])
                s_itr  = s_itr[:, :K_safe]
                s_ival = pca.transform(X_ival_c)[:, :K_safe]

                clf = SVC(kernel=kernel, C=C, class_weight=cw)
                clf.fit(s_itr, y_itr)
                inner_preds.extend((clf.decision_function(s_ival) > 0).astype(int))
                inner_true.extend(y_ival)

            score = balanced_accuracy_score(inner_true, inner_preds)
            if score > best_inner_score:
                best_inner_score = score
                best_params = {'K': K_safe, 'C': C, 'kernel': kernel, 'class_weight': cw}

        # ---- Outer: retrain on full training split with best (K, C, kernel, cw) ----
        mean_tr  = X_tr.mean(axis=0)
        X_tr_c   = X_tr - mean_tr
        X_te_c   = X_te - mean_tr

        pca   = PCA()
        s_tr  = pca.fit_transform(X_tr_c)

        # ← CHANGED: use best_params['K'] from inner search, not a variance threshold
        K_best = best_params['K']
        s_tr   = s_tr[:, :K_best]
        s_te   = pca.transform(X_te_c)[:, :K_best]

        clf = SVC(C=best_params['C'], kernel=best_params['kernel'],
                  class_weight=best_params['class_weight'])
        clf.fit(s_tr, y_tr)
        decision = clf.decision_function(s_te)[0]

        y_pred.append((decision > 0).astype(int))
        y_true.append(y_te[0])
        decisions.append(decision)
        best_params_per_fold.append(best_params)

        print(f"Fold {fold+1}: best_params={best_params}, inner_BA={best_inner_score:.4f}")

    results = {
        "y_true":               np.array(y_true),
        "y_pred":               np.array(y_pred),
        "decisions":            np.array(decisions),
        "balanced_accuracy":    balanced_accuracy_score(y_true, y_pred),
        "best_params_per_fold": best_params_per_fold
    }

    print(f"\nOuter LOOCV Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    return results
