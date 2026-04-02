"""
Sophisticated SVM Implementation for Breath Analysis
Based on Maiti et al. 2021 methodology with tuned hyperparameters
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional

HYPERPARAMS_PATH = '/home/anushkasingh/Desktop/Thesis/Code/classical_SVM_pipeline'


class SVMBreathClassifier:
    """
    SVM classifier for breath analysis following Maiti et al. 2021 methodology
    """

    def __init__(self, hyperparams_path: str = HYPERPARAMS_PATH):
        """
        Initialize classifier with tuned hyperparameters

        Args:
            hyperparams_path: Path to CSV files with tuned hyperparameters
        """
        self.hyperparams_path = hyperparams_path
        self.best_params = self._load_hyperparameters()

    def _load_hyperparameters(self) -> Dict:
        """
        Load best hyperparameters from CSV files.
        Loads from all_configs_best_params.csv (primary, from systematic search),
        then exp1/exp2 CSVs as fallback if available.
        All keys are unified as (config_id_or_sr, task).
        """
        params = {}

        # --- Primary: all_configs_best_params.csv (72-config systematic search) ---
        all_configs_path = f'{self.hyperparams_path}/all_configs_best_params.csv'
        try:
            all_configs = pd.read_csv(all_configs_path)
            for _, row in all_configs.iterrows():
                if row.get('status') != 'done':
                    continue
                key = (row['config_id'], row['task'])
                params[key] = {
                    'sigma':            row['best_sigma'],
                    'kernel':           row['best_kernel'],
                    'C':                row['best_C'],
                    'gamma':            row['best_gamma'],
                    'degree':           row['best_degree'] if row['best_kernel'] == 'poly' else None,
                    'feature_type':     row['feature_type'],
                    'n_pca_components': int(row['n_pca_components'])
                                        if pd.notna(row.get('n_pca_components')) else 4,
                    'sr_mode':          row['sr_mode'],
                    'sr_col':           row['sr_col'],
                }
        except FileNotFoundError:
            pass

        # --- Fallback: exp1 single SR comparison ---
        try:
            exp1 = pd.read_csv(f'{self.hyperparams_path}/exp1_single_sr_comparison.csv')
            for _, row in exp1.iterrows():
                key = (row['SR'], row['Task'])
                params[key] = {
                    'sigma':            row['Sigma'],
                    'kernel':           row['Kernel'],
                    'C':                row['C'],
                    'gamma':            row['Gamma'],
                    'degree':           row['Degree'] if row['Kernel'] == 'poly' else None,
                    'feature_type':     row['Feature_type'],
                    'n_pca_components': 4,  # exp1 is whole_sr, PCA not used
                }
        except FileNotFoundError:
            pass

        # --- Fallback: exp2 whole_sr vs PCA ---
        try:
            exp2 = pd.read_csv(f'{self.hyperparams_path}/exp2_whole_vs_pca.csv')
            for _, row in exp2.iterrows():
                key = (row['Method'], row['Task'])
                params[key] = {
                    'sigma':            row['Sigma'],
                    'kernel':           row['Kernel'],
                    'C':                row['C'],
                    'gamma':            row['Gamma'],
                    'degree':           row['Degree'] if row['Kernel'] == 'poly' else None,
                    'feature_type':     row['Feature_type'],
                    'n_pca_components': 4,
                }
        except FileNotFoundError:
            pass

        return params

    def _apply_smoothing(self, X: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian smoothing to each sample. sigma=0 means no smoothing.
        SR data is already mean-centered + normalized by sr_preprocessing.py,
        so no additional normalization is applied here.

        Args:
            X: Input data (n_samples, n_features)
            sigma: Gaussian filter width

        Returns:
            Smoothed data
        """
        # Guard against NaN/None sigma from CSV loading
        if sigma is None or (isinstance(sigma, float) and np.isnan(sigma)) or sigma == 0:
            return X.copy()
        return np.array([gaussian_filter1d(x, sigma) for x in X])

    def _create_svm(self, params: Dict) -> SVC:
        """
        Create SVM classifier with given parameters

        Args:
            params: Dictionary of SVM parameters

        Returns:
            Configured SVM classifier
        """
        svm_params = {
            'C': params['C'],
            'kernel': params['kernel'],
            # 'class_weight': 'balanced',  # match grid_search.py — handles class imbalance
            'random_state': 42,
            'max_iter': -1
        }

        if params['kernel'] in ['rbf', 'poly']:
            svm_params['gamma'] = params['gamma']

        if params['kernel'] == 'poly' and params['degree'] is not None:
            svm_params['degree'] = int(params['degree'])

        return SVC(**svm_params)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute classification metrics following paper definitions.
        For multi-class tasks (e.g. H_vs_KC_BC), sensitivity and specificity
        are computed as H (healthy=negative) vs all cancer (positive).

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        classes = np.unique(y_true)

        if len(classes) == 2:
            # Binary: standard 2x2 confusion matrix
            # H = healthy (negative), cancer = positive
            pos_label = classes[classes != 'H'][0] if 'H' in classes else classes[1]
            cm = confusion_matrix(y_true, y_pred, labels=['H', pos_label])
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # cancer recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # healthy recall
            return {
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'confusion_matrix': cm,
                'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            }
        else:
            # Multi-class: collapse to H vs all-cancer for sensitivity/specificity
            y_true_bin = np.where(y_true == 'H', 'H', 'cancer')
            y_pred_bin = np.where(y_pred == 'H', 'H', 'cancer')
            cm = confusion_matrix(y_true_bin, y_pred_bin, labels=['H', 'cancer'])
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            return {
                'accuracy': accuracy,
                'sensitivity': sensitivity,   # H vs all-cancer
                'specificity': specificity,
                'confusion_matrix': cm,
                'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            }

    def loocv_validation(self, X: np.ndarray, y: np.ndarray, params: Dict) -> Dict:
        """
        Leave-One-Out Cross Validation (Table 1, columns 7-9 in paper)
        PCA is fitted INSIDE each fold to avoid data leakage.

        Args:
            X: Feature matrix (already SR-preprocessed)
            y: Labels
            params: SVM parameters

        Returns:
            LOOCV results with accuracy, sensitivity, specificity
        """
        X_proc = self._apply_smoothing(X, params['sigma'])
        use_pca = params.get('feature_type') == 'pca'
        n_pca = params.get('n_pca_components', 4)  # read from params, not hardcoded

        loo = LeaveOneOut()
        y_pred_all = np.empty(len(y), dtype=object)
        decisions = np.zeros(len(y))  # continuous scores for gray zone analysis

        for train_idx, test_idx in loo.split(X_proc):
            X_tr, X_te = X_proc[train_idx], X_proc[test_idx]
            y_tr = y[train_idx]

            # PCA fitted on training fold only — no leakage
            if use_pca:
                pca = PCA(n_components=n_pca)
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)

            clf = self._create_svm(params)
            clf.fit(X_tr, y_tr)
            y_pred_all[test_idx] = clf.predict(X_te)
            dec = clf.decision_function(X_te)
            decisions[test_idx] = dec if dec.ndim == 1 else dec.max(axis=1)

        metrics = self._compute_metrics(y, y_pred_all)
        metrics['y_pred'] = y_pred_all
        metrics['decisions'] = decisions
        metrics['method'] = 'LOOCV'
        return metrics

    def kfold_validation(self, X: np.ndarray, y: np.ndarray, params: Dict,
                         k: int, n_repeats: int = 10) -> Dict:
        """
        Repeated Stratified K-Fold Cross Validation (Table 1 in paper)
        n_repeats=10 matches Maiti 2021 methodology.
        PCA is fitted INSIDE each fold to avoid data leakage.

        Args:
            X: Feature matrix
            y: Labels
            params: SVM parameters
            k: Number of folds
            n_repeats: Number of repetitions (paper uses 10)

        Returns:
            K-fold results with mean and std of metrics
        """
        X_proc = self._apply_smoothing(X, params['sigma'])
        use_pca = params.get('feature_type') == 'pca'
        n_pca = params.get('n_pca_components', 4)  # read from params, not hardcoded

        from sklearn.metrics import matthews_corrcoef as _mcc

        accuracies = []
        sensitivities = []
        specificities = []
        mccs = []

        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42 + repeat)
            y_pred_fold = np.empty(len(y), dtype=object)

            for train_idx, test_idx in skf.split(X_proc, y):
                X_tr, X_te = X_proc[train_idx], X_proc[test_idx]
                y_tr = y[train_idx]

                # PCA fitted on training fold only — no leakage
                if use_pca:
                    pca = PCA(n_components=n_pca)
                    X_tr = pca.fit_transform(X_tr)
                    X_te = pca.transform(X_te)

                clf = self._create_svm(params)
                clf.fit(X_tr, y_tr)
                y_pred_fold[test_idx] = clf.predict(X_te)

            m = self._compute_metrics(y, y_pred_fold)
            accuracies.append(m['accuracy'])
            sensitivities.append(m['sensitivity'])
            specificities.append(m['specificity'])
            mccs.append(_mcc(y, y_pred_fold))

        return {
            'accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'sensitivity': np.mean(sensitivities),
            'sensitivity_std': np.std(sensitivities),
            'specificity': np.mean(specificities),
            'specificity_std': np.std(specificities),
            'mcc': np.mean(mccs),
            'mcc_std': np.std(mccs),
            'balanced_accuracy': (np.mean(sensitivities) + np.mean(specificities)) / 2,
            'method': f'{k}-fold (x{n_repeats})',
            'n_repeats': n_repeats,
        }

    def comprehensive_evaluation(self, X: np.ndarray, y: np.ndarray,
                                  data_type: str, task: str,
                                  k_values: List[int] = [9]) -> pd.DataFrame:
        """
        Comprehensive evaluation with LOOCV and multiple k-fold validations.
        Looks up best params from loaded CSVs using (data_type, task) key.

        Args:
            X: Feature matrix — must already be filtered to the correct task's samples
            y: Labels — must match X rows (same task filter applied)
            data_type: config_id from all_configs_best_params.csv, or SR column name,
                       or method name ('whole_sr', 'PCA_top4')
            task: Classification task (e.g. 'H_vs_PC', 'H_vs_KC')
            k_values: List of k values for k-fold validation (default [9] = paper)

        Returns:
            DataFrame with all results
        """
        param_key = (data_type, task)
        if param_key not in self.best_params:
            raise KeyError(f"No params found for {param_key}. "
                           f"Available keys (first 5): {list(self.best_params.keys())[:5]}")

        params = self.best_params[param_key]
        results = []

        from scipy.stats import beta as beta_dist
        from sklearn.metrics import matthews_corrcoef

        def clopper_pearson(k, n, alpha=0.05):
            lo = beta_dist.ppf(alpha/2,   k,   n-k+1) if k > 0 else 0.0
            hi = beta_dist.ppf(1-alpha/2, k+1, n-k)   if k < n else 1.0
            return round(float(lo), 4), round(float(hi), 4)

        # LOOCV (primary method in paper)
        print(f"  LOOCV: {task} | {data_type} ...")
        loocv_result = self.loocv_validation(X, y, params)
        tp = loocv_result['TP']
        tn = loocv_result['TN']
        fn = loocv_result['FN']
        fp = loocv_result['FP']
        mcc = matthews_corrcoef(y, loocv_result['y_pred'])
        sens_ci = clopper_pearson(tp, tp + fn)
        spec_ci = clopper_pearson(tn, tn + fp)
        results.append({
            'Task': task, 'Data_Type': data_type, 'Method': 'LOOCV',
            'Accuracy': loocv_result['accuracy'],        'Accuracy_std': 0.,
            'Sensitivity': loocv_result['sensitivity'],  'Sensitivity_std': 0.,
            'Specificity': loocv_result['specificity'],  'Specificity_std': 0.,
            'MCC': mcc,                                  'MCC_std': 0.,
            'Sens_CI_lo': sens_ci[0], 'Sens_CI_hi': sens_ci[1],
            'Spec_CI_lo': spec_ci[0], 'Spec_CI_hi': spec_ci[1],
            **{k: v for k, v in params.items() if k != 'feature_type'},
        })

        # K-fold validations
        for k in k_values:
            print(f"  {k}-fold: {task} | {data_type} ...")
            kfold_result = self.kfold_validation(X, y, params, k)
            results.append({
                'Task': task, 'Data_Type': data_type, 'Method': kfold_result['method'],
                'Accuracy': kfold_result['accuracy'],
                'Accuracy_std': kfold_result['accuracy_std'],
                'Sensitivity': kfold_result['sensitivity'],
                'Sensitivity_std': kfold_result['sensitivity_std'],
                'Specificity': kfold_result['specificity'],
                'Specificity_std': kfold_result['specificity_std'],
                'MCC': kfold_result.get('mcc', None),
                'MCC_std': kfold_result.get('mcc_std', None),
                'Sens_CI_lo': None, 'Sens_CI_hi': None,
                'Spec_CI_lo': None, 'Spec_CI_hi': None,
                **{k2: v for k2, v in params.items() if k2 != 'feature_type'},
            })

        return pd.DataFrame(results)

    def blind_set_evaluation(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             params: Dict, threshold: float = 0.0) -> Dict:
        """
        Blind set evaluation (Table 2 in paper).
        Train on training set, evaluate on truly independent blind set.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Blind set features
            y_test: Blind set labels
            params: SVM parameters
            threshold: Decision threshold. SVM decision_function is centred at 0
                       (positive = class[1], negative = class[0]), so default is 0.
                       Shift positive (e.g. 0.3) to increase specificity at cost of sensitivity.

        Returns:
            Blind set results with sensitivity, specificity at given threshold
        """
        X_tr = self._apply_smoothing(X_train, params['sigma'])
        X_te = self._apply_smoothing(X_test, params['sigma'])
        n_pca = params.get('n_pca_components', 4)  # read from params, not hardcoded

        # PCA fitted on training set only — no leakage into blind set
        if params.get('feature_type') == 'pca':
            pca = PCA(n_components=n_pca)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = self._create_svm(params)
        clf.fit(X_tr, y_train)

        # Get decision function for threshold-based classification.
        # Decision boundary is at 0 for binary SVM — threshold shifts this.
        # sklearn sorts classes_ alphabetically, so classes_[1] is not always cancer.
        # Standardise sign so that positive score always means cancer (non-H).
        decisions = clf.decision_function(X_te)
        classes = clf.classes_
        cancer_label = classes[classes != 'H'][0]  # always the non-H class
        healthy_label = 'H'
        # If H is classes_[1], the sign is flipped relative to our convention — correct it.
        if classes[1] == 'H':
            decisions = -decisions
        y_pred = np.where(decisions > threshold, cancer_label, healthy_label)

        metrics = self._compute_metrics(y_test, y_pred)
        metrics['decisions'] = decisions
        metrics['threshold'] = threshold
        return metrics


def run_full_analysis(X_dict: Dict[str, np.ndarray], y_full: np.ndarray,
                      tasks: List[str],
                      task_classes: Dict[str, List[str]],
                      output_path: str = HYPERPARAMS_PATH) -> pd.DataFrame:
    """
    Run complete analysis pipeline for all configurations.

    Args:
        X_dict: Dictionary of feature matrices e.g.
                {'SR_1005_preprocessed': X1, 'whole_sr': X2, 'PCA_top4': X3}
                Each X must contain rows for ALL patients (unfiltered).
        y_full: Full label array for all patients (unfiltered).
        tasks: List of classification tasks e.g. ['H_vs_PC', 'H_vs_KC']
        task_classes: Dict mapping task name to the two classes involved e.g.
                      {'H_vs_PC': ['H', 'PC'], 'H_vs_KC_BC': ['H', 'KC', 'BC']}
                      Used to filter X and y correctly per task.
        output_path: Path to save results CSV

    Returns:
        DataFrame with all results
    """
    classifier = SVMBreathClassifier()
    all_results = []

    for data_type, X_all in X_dict.items():
        for task in tasks:
            print(f"\n{'='*60}")
            print(f"Analyzing: {task} with {data_type}")
            print(f"{'='*60}")

            # Filter X and y to only the classes involved in this task
            classes = task_classes.get(task, [])
            if classes:
                mask = np.isin(y_full, classes)
                X = X_all[mask]
                y = y_full[mask]
            else:
                X, y = X_all, y_full

            try:
                results_df = classifier.comprehensive_evaluation(X, y, data_type, task)
                all_results.append(results_df)
            except KeyError as e:
                print(f"  SKIPPED: {e}")

    # Combine all results
    final_results = pd.concat(all_results, ignore_index=True)

    # Save results
    output_file = f'{output_path}/svm_comprehensive_results.csv'
    final_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Print summary (paper format - Table 1)
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY (Following Paper Table 1 Format)")
    print("="*80)

    for task in tasks:
        print(f"\n{task}:")
        task_results = final_results[final_results['Task'] == task]

        for data_type in X_dict.keys():
            dt_results = task_results[task_results['Data_Type'] == data_type]
            if not dt_results.empty:
                print(f"\n  {data_type}:")
                for _, row in dt_results.iterrows():
                    print(f"    {row['Method']:15s}: Acc={row['Accuracy']:.3f} "
                          f"Sens={row['Sensitivity']:.3f} Spec={row['Specificity']:.3f} "
                          f"SD={row['Std']:.3f}")

    return final_results
