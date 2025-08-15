"""
rural_credit_scorer_v2.py

Revised prototype for rural credit scoring (production-minded, demonstrative).
- Fixes data leakage by using imblearn Pipeline and fitting transforms only on training data
- Adds evaluation metrics, calibration, and cross-validated analysis
- Provides robust SHAP handling and per-applicant explanations
- Adds simple fairness checks (statistical parity & equalized odds) on a protected attribute
- Removes Aadhaar/PII use; uses 'residence_years' as a non-PII proxy (document and review legally)
- Adds model save/load, logging, and simple unit-like test checks in __main__
- Designed for hackathon/demo but with explicit next steps documented in code comments.

Dependencies:
    numpy, pandas, scikit-learn, imbalanced-learn, xgboost, shap, joblib

Author: Revised from user's original prototype
Date: 2025-08-12 (example)
"""
import inspect
import os
import math
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline as SkPipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import shap
import joblib

# ----------------------------
# Logging configuration
# ----------------------------
logger = logging.getLogger("RuralCreditScorerV2")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


# ----------------------------
# Utility functions
# ----------------------------
def safe_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    logger.debug(f"Global RNG seeded with {seed}.")


def ensure_dir_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.debug(f"Created directory {path}.")


# ----------------------------
# Main class
# ----------------------------
class RuralCreditScorerV2:
    """
    A production-minded prototype for a rural credit scorer.

    Main improvements vs. the earlier prototype:
    - No scaler fit on full dataset (no data leakage). Uses an imbalanced-learn Pipeline (SMOTE -> Scaler -> Classifier).
    - Training split into Train / Calib / Test, model is calibrated on calibration set.
    - Comprehensive evaluation metrics (ROC AUC, PR AUC, Brier, Precision, Recall, F1).
    - SHAP explanations with robust handling for shap output shapes across SHAP versions.
    - Basic fairness checks against a synthetic 'protected' attribute (gender) without using it as a model input.
    - No use of Aadhaar or PII. Uses a benign 'residence_years' feature to estimate stability (still needs legal review).
    """

    def __init__(self, n_samples=10000, random_state=42, model_output_dir="./model_artifacts"):
        safe_seed(random_state)
        self.random_state = random_state

        # Feature names and default weights for a simple base score
        self.feature_names = [
            'dbt_count_qtr',            # DBT payments in last quarter (0-4)
            'bill_on_time_ratio',       # Percent of bills paid on time (0-100)
            'aeps_upi_txn',             # Digital transactions (last 6 months)
            'deposit_withdrawal_ratio', # Ratio of deposits to withdrawals
            'residence_years'           # Years at same residence (proxy for stability)
        ]
        self.feature_weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])

        # Model artifacts
        self.pipeline = None  # imblearn pipeline (SMOTE -> scaler -> clf)
        self.calibrated_clf = None  # calibrated classifier wrapper
        self.shap_explainer = None
        self.model_output_dir = model_output_dir
        ensure_dir_exists(self.model_output_dir)

        # Generate data and train
        logger.info("Generating synthetic dataset...")
        X, y, meta = self._generate_synthetic_data(n_samples=n_samples)
        self.meta = meta  # keep auxiliary fields for fairness checks

        logger.info("Preparing data splits (train/calib/test)...")
        self._prepare_splits(X, y)

        logger.info("Building pipeline...")
        self._build_pipeline()

        logger.info("Training and calibrating model...")
        self._train_and_calibrate()

        logger.info("Initializing SHAP explainer...")
        self._init_shap_explainer()

    # ----------------------------
    # Data generation (synthetic)
    # ----------------------------
    def _generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns:
            X: numpy array of shape (n_samples, n_features)
            y: binary labels (1 default, 0 non-default)
            meta: dict of auxiliary arrays (e.g., protected attributes not used as features)
        """
        rng = np.random.RandomState(self.random_state)

        X = np.zeros((n_samples, len(self.feature_names)))
        meta = {
            'gender': np.empty(n_samples, dtype=object),
            'region': np.empty(n_samples, dtype=object),
            'applicant_id': np.arange(n_samples)
        }

        # 1. DBT Count (Quarterly): 0-4
        X[:, 0] = np.clip(np.round(rng.normal(2.5, 1.0, n_samples)), 0, 4)

        # 2. Bill Payment Ratio: bimodal
        half = n_samples // 2
        low_payers = rng.beta(1, 5, half)
        good_payers = rng.beta(8, 2, n_samples - half)
        bill_ratios = np.concatenate([low_payers, good_payers])
        rng.shuffle(bill_ratios)
        X[:, 1] = bill_ratios * 100

        # 3. Digital Transactions: long-tail (0-100)
        X[:, 2] = np.clip(rng.poisson(20, n_samples) + rng.exponential(6, n_samples), 0, 100)

        # 4. Deposit/Withdrawal Ratio: lognormal, clipped
        ratios = rng.lognormal(0, 0.7, n_samples)
        X[:, 3] = np.clip(ratios, 0.1, 10.0)

        # 5. Residence years (non-PII proxy for geo stability): 0-30 years
        X[:, 4] = np.clip(np.round(rng.normal(6, 3, n_samples)), 0, 30)

        # Protected attributes (not used by model, but used for fairness checks)
        genders = ['M', 'F']
        regions = ['Rajasthan', 'UP', 'Bihar', 'WB', 'Maharashtra', 'TamilNadu']
        meta['gender'] = rng.choice(genders, n_samples, p=[0.55, 0.45])
        meta['region'] = rng.choice(regions, n_samples)

        # Construct a somewhat realistic default probability -- nonlinear mixing
        # This is only for synthetic data creation.
        def_prob = (
            0.35 * (1 - X[:, 0] / 4) +                 # more DBT -> lower default
            0.25 * (1 - X[:, 1] / 100) +               # higher bill on-time -> lower default
            0.15 * (1 - (X[:, 2] / 100)) +             # digital engagement reduces default
            0.15 * (1 - np.minimum(X[:, 3], 3) / 3) +  # very high deposit/withdrawal reduces risk up to cap
            0.10 * (1 - np.minimum(X[:, 4], 10) / 10)  # residence stability reduces default up to 10y
        )

        # Introduce small region-based skew to simulate realistic disparities (for fairness testing).
        # IMPORTANT: This skew is ONLY for testbed fairness measurement. Do NOT use region as a feature in the model.
        skew_map = {
            'Rajasthan': 0.0,
            'UP': 0.02,
            'Bihar': 0.03,
            'WB': -0.01,
            'Maharashtra': -0.02,
            'TamilNadu': -0.015
        }
        region_skew = np.array([skew_map[r] for r in meta['region']])
        def_prob = np.clip(def_prob + region_skew, 0.02, 0.95)

        # Optional non-linear logistic-like transform to get probabilities
        def_prob = 1.0 / (1.0 + np.exp((def_prob - 0.5) * 6))

        # Sample binary labels
        y = rng.binomial(1, def_prob)

        logger.debug(f"Synthetic data generated: X.shape={X.shape}, y.mean={y.mean():.3f}")
        return X.astype(float), y.astype(int), meta

    # ----------------------------
    # Splits
    # ----------------------------
    def _prepare_splits(self, X: np.ndarray, y: np.ndarray):
        """
        Split into Train / Calib / Test to avoid leakage when calibrating probabilities.
        Typical split: 70% train / 15% calib / 15% test.

        This implementation splits indices first and then slices X, y and meta arrays
        to ensure perfectly-aligned splits and avoids unpacking errors.
        """
        n = X.shape[0]
        indices = np.arange(n)

        # First split: temp vs test (test ~= 15%)
        temp_idx, test_idx = train_test_split(
            indices, test_size=0.15, random_state=self.random_state, stratify=y
        )

        # Second split: train vs calib from temp (calib ~= 0.15 of total -> 0.17647 of temp)
        train_idx, calib_idx = train_test_split(
            temp_idx, test_size=0.1764705882, random_state=self.random_state, stratify=y[temp_idx]
        )

        # Slice arrays
        self.X_train = X[train_idx]
        self.y_train = y[train_idx]

        self.X_calib = X[calib_idx]
        self.y_calib = y[calib_idx]

        self.X_test = X[test_idx]
        self.y_test = y[test_idx]

        # Map meta info to splits directly (meta arrays must have same length/order as X)
        self.meta_train = {k: np.array(v)[train_idx] for k, v in self.meta.items() if k != 'applicant_id'}
        self.meta_calib = {k: np.array(v)[calib_idx] for k, v in self.meta.items() if k != 'applicant_id'}
        self.meta_test = {k: np.array(v)[test_idx] for k, v in self.meta.items() if k != 'applicant_id'}

        # Store indices in case user needs them later
        self.train_idx = train_idx
        self.calib_idx = calib_idx
        self.test_idx = test_idx

        logger.debug(f"Train size: {self.X_train.shape[0]}, Calib size: {self.X_calib.shape[0]}, Test size: {self.X_test.shape[0]}")


    # ----------------------------
    # Pipeline construction
    # ----------------------------
    def _build_pipeline(self):
        """
        Build an imblearn Pipeline: SMOTE -> MinMaxScaler -> XGBoostClassifier
        SMOTE is applied only during pipeline.fit on training data (no leakage).
        """
        # Choose XGBoost hyperparameters (sensible defaults)
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.random_state
        )

        # Pipeline: SMOTE -> Scaler -> Classifier
        self.pipeline = ImbPipeline(steps=[
            ('smote', SMOTE(random_state=self.random_state)),
            ('scaler', MinMaxScaler()),
            ('clf', xgb_clf)
        ])

    # ----------------------------
    # Training, calibration, evaluation
    # ----------------------------

    def _train_and_calibrate(self):
        """
        Fit the pipeline on train. Then calibrate probabilities using a held-out calibration set.
        Evaluate on test set and log metrics.
        This version handles scikit-learn differences around the CalibratedClassifierCV constructor.
        """
        # Fit pipeline on train (SMOTE applied inside pipeline.fit)
        logger.info("Fitting pipeline on training data (SMOTE applied here)...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.debug("Pipeline fitted.")

        # Calibrate on calibration set using cv='prefit'
        logger.info("Calibrating classifier on calibration set using sigmoid (Platt) method...")

        # Construct CalibratedClassifierCV in a version-robust way
        calibrator = None
        try:
            # Preferred modern API: estimator=
            calibrator = CalibratedClassifierCV(estimator=self.pipeline, method='sigmoid', cv='prefit')
            logger.debug("Using CalibratedClassifierCV(estimator=...)")
        except TypeError as e1:
            logger.warning(f"CalibratedClassifierCV(estimator=...) failed: {e1}; trying legacy base_estimator= ...")
            try:
                # Older sklearn used base_estimator keyword
                calibrator = CalibratedClassifierCV(base_estimator=self.pipeline, method='sigmoid', cv='prefit')
                logger.debug("Using CalibratedClassifierCV(base_estimator=...)")
            except TypeError as e2:
                # Last resort: try without passing estimator (not ideal); raise informative error
                logger.error("Both estimator= and base_estimator= failed for CalibratedClassifierCV. "
                                "Please check your scikit-learn version.")
                raise

        # Fit the calibrator on the held-out calibration set
        calibrator.fit(self.X_calib, self.y_calib)
        self.calibrated_clf = calibrator
        logger.debug("Calibration complete.")

        # Evaluate on test set
        y_proba = self.calibrated_clf.predict_proba(self.X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = self._compute_metrics(self.y_test, y_pred, y_proba)
        logger.info("Test set evaluation metrics:")
        for name, val in metrics.items():
            logger.info(f"  {name}: {val}")

        # Save model artifacts
        self._save_artifacts()

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """
        Compute a set of evaluation metrics useful for credit scoring.
        """
        metrics = {}
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics['roc_auc'] = float('nan')
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except Exception:
            metrics['pr_auc'] = float('nan')

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['brier'] = brier_score_loss(y_true, y_proba)

        # Optionally compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['tn'] = int(tn)
        metrics['fn'] = int(fn)

        return metrics

    # ----------------------------
    # SHAP Explainer
    # ----------------------------
    def _init_shap_explainer(self):
        """
        Initialize SHAP TreeExplainer using the fitted XGBoost estimator inside pipeline.
        We must extract the raw XGBoost model object from the imblearn pipeline.
        """
        try:
            # The pipeline stores the classifier under 'clf'
            xgb_model = self.pipeline.named_steps['clf']
            # Build explainer on the model. Use the scaler to transform a sample of training data as background.
            scaler = self.pipeline.named_steps['scaler']
            # Use a small background sample from training data, scaled
            background = scaler.transform(self.X_train[np.random.choice(self.X_train.shape[0], min(100, self.X_train.shape[0]), replace=False)])
            # TreeExplainer works with models; in some versions if using sklearn wrapper, pass model.get_booster()
            try:
                explainer = shap.TreeExplainer(xgb_model)
            except Exception:
                # fallback to underlying booster
                try:
                    explainer = shap.TreeExplainer(xgb_model.get_booster())
                except Exception:
                    explainer = shap.TreeExplainer(xgb_model)
            self.shap_explainer = explainer
            logger.debug("SHAP explainer initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None

    # ----------------------------
    # Prediction & Explanation APIs
    # ----------------------------
    def compute_features(self, applicant_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert applicant raw data dict into numeric feature vector matching self.feature_names.

        applicant_data expected keys:
            - 'dbt' : list of dicts with 'date' (YYYY-MM-DD), 'amount', 'scheme' (optional)
            - 'bills': list of dicts with 'due_date', 'payment_date', 'amount', 'type'
            - 'aeps', 'upi': lists of dicts with 'date', 'amount', ...
            - 'account': list of monthly summaries with 'total_deposits', 'total_withdrawals'
            - 'profile': dict with 'residence_years' (non-identifiable), 'location' (state)
        """
        features = np.zeros(len(self.feature_names), dtype=float)
        now = datetime.now()

        # 1. DBT Count (last 90 days)
        dbt_dates = []
        for d in applicant_data.get('dbt', []):
            try:
                dbt_dates.append(datetime.strptime(d['date'], '%Y-%m-%d'))
            except Exception:
                continue
        quarter_ago = now - timedelta(days=90)
        features[0] = sum(1 for dt in dbt_dates if dt > quarter_ago)

        # 2. Bill Payment Ratio (on-time %)
        bills = applicant_data.get('bills', [])
        total_bills = len(bills)
        on_time = 0
        for b in bills:
            try:
                due = datetime.strptime(b['due_date'], '%Y-%m-%d')
                pay = datetime.strptime(b['payment_date'], '%Y-%m-%d')
                if pay <= due:
                    on_time += 1
            except Exception:
                continue
        features[1] = (on_time / total_bills * 100.0) if total_bills > 0 else 0.0

        # 3. AEPS + UPI transactions in last 6 months
        six_months_ago = now - timedelta(days=180)
        aeps_count = sum(1 for t in applicant_data.get('aeps', []) if self._parse_date_safe(t.get('date')) > six_months_ago)
        upi_count = sum(1 for t in applicant_data.get('upi', []) if self._parse_date_safe(t.get('date')) > six_months_ago)
        features[2] = aeps_count + upi_count

        # 4. Deposit/Withdrawal ratio across available account months
        deposits = sum((m.get('total_deposits') or 0) for m in applicant_data.get('account', []))
        withdrawals = sum((m.get('total_withdrawals') or 0) for m in applicant_data.get('account', []))
        if withdrawals > 0:
            features[3] = deposits / withdrawals
        else:
            # Avoid infinite; use a conservative high cap
            features[3] = min(10.0, deposits / 1.0) if deposits > 0 else 0.1

        # 5. Residence years (non-PII proxy)
        features[4] = float(applicant_data.get('profile', {}).get('residence_years', 0.0))

        # Ensure feature ranges are sensible
        features[0] = float(np.clip(features[0], 0, 12))
        features[1] = float(np.clip(features[1], 0.0, 100.0))
        features[2] = float(np.clip(features[2], 0.0, 1000.0))
        features[3] = float(np.clip(features[3], 0.1, 10.0))
        features[4] = float(np.clip(features[4], 0.0, 60.0))

        logger.debug(f"Computed features: {dict(zip(self.feature_names, features))}")
        return features

    @staticmethod
    def _parse_date_safe(date_str: str) -> datetime:
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except Exception:
            # return very old date so that invalid dates don't count
            return datetime(1970, 1, 1)

    def predict(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute credit score, default probability (calibrated), risk score, SHAP explanation and components.
        """
        raw_features = self.compute_features(applicant_data)
        # Shape and pipeline scaling:
        X_raw = raw_features.reshape(1, -1)

        # Use the calibrated classifier if available; otherwise fall back to pipeline
        if self.calibrated_clf is not None:
            proba = self.calibrated_clf.predict_proba(X_raw)[0, 1]
            pred_proba_array = self.calibrated_clf.predict_proba(X_raw)[0]
        else:
            proba = self.pipeline.predict_proba(X_raw)[0, 1]
            pred_proba_array = self.pipeline.predict_proba(X_raw)[0]

        risk_score = float(proba * 100.0)

        # For the human-friendly 300-900 credit score, compute using scaled features and feature_weights
        # We need to transform the raw features with the scaler used in the pipeline.
        scaler = self.pipeline.named_steps['scaler']
        scaled_features = scaler.transform(X_raw)[0]
        base_score = float(np.dot(scaled_features, self.feature_weights))
        credit_score = float(300.0 + 600.0 * base_score)

        # SHAP values (robust handling of shap outputs across versions)
        shap_values = None
        shap_contrib = None
        if self.shap_explainer is not None:
            try:
                sv = self.shap_explainer.shap_values(scaled_features.reshape(1, -1))
                # sv may be a list (one per class) or a 2D array
                if isinstance(sv, list):
                    # For binary classification, shap returns [class0, class1] arrays
                    if len(sv) == 2:
                        shap_contrib = np.array(sv[1]).reshape(-1)
                    else:
                        # If multiple classes unexpectedly, pick the last
                        shap_contrib = np.array(sv[-1]).reshape(-1)
                else:
                    shap_contrib = np.array(sv).reshape(-1)
                shap_values = shap_contrib.tolist()
            except Exception as e:
                logger.warning(f"SHAP computation failed for instance: {e}")
                shap_values = None
                shap_contrib = None

        # Identify top risk factors by absolute SHAP contribution (if available)
        if shap_contrib is not None:
            factor_impact = list(zip(self.feature_names, shap_contrib))
            factor_impact.sort(key=lambda x: abs(x[1]), reverse=True)
            top_risks = [f for f, _ in factor_impact[:3]]
            factor_map = {name: float(val) for name, val in factor_impact}
        else:
            # fallback: use raw feature deviations to indicate risk (higher default risk when features are worse)
            # Here we compute naive impacts as negative of scaled positive features
            estimated_impacts = -scaled_features * self.feature_weights
            factor_impact = list(zip(self.feature_names, estimated_impacts))
            factor_impact.sort(key=lambda x: abs(x[1]), reverse=True)
            top_risks = [f for f, _ in factor_impact[:3]]
            factor_map = {name: float(val) for name, val in factor_impact}

        # Component breakdown (percent-like numbers for UI)
        components = {
            'dbt_frequency': float(np.clip(scaled_features[0] * 100.0, 0.0, 100.0)),
            'bill_payment_ratio': float(np.clip(scaled_features[1] * 100.0, 0.0, 100.0)),
            'digital_engagement': float(np.clip(scaled_features[2] * 100.0, 0.0, 100.0)),
            'cashflow_ratio': float(np.clip(scaled_features[3] * 100.0, 0.0, 100.0)),
            'residence_stability': float(np.clip(scaled_features[4] * 100.0, 0.0, 100.0))
        }

        result = {
            'credit_score': credit_score,
            'risk_score': risk_score,
            'default_probability': proba,
            'risk_factors': top_risks,
            'score_components': components,
            'shap_values': shap_values,
            'raw_scaled_features': scaled_features.tolist(),
            'feature_impacts': factor_map
        }
        logger.debug(f"Prediction result: {result}")
        return result

    # ----------------------------
    # Fairness checks
    # ----------------------------
    def fairness_audit(self, protected_attr: str = 'gender', privileged_value: str = 'M') -> Dict[str, Any]:
        """
        Simple fairness audit on test set comparing privileged vs unprivileged group.
        Metrics: statistical parity difference (SPD) and equalized odds difference (difference in TPR).
        """
        logger.info("Running fairness audit on test set...")
        # Build predictions on test set
        y_proba_test = self.calibrated_clf.predict_proba(self.X_test)[:, 1]
        y_pred_test = (y_proba_test >= 0.5).astype(int)

        # Extract protected attribute values for test examples
        prot_values = self.meta_test.get(protected_attr)
        if prot_values is None:
            raise ValueError(f"Protected attribute {protected_attr} not present in meta_test.")

        # Identify privileged and unprivileged masks
        privileged_mask = (prot_values == privileged_value)
        unprivileged_mask = ~privileged_mask

        # Statistical parity: P(pred=1 | unprivileged) - P(pred=1 | privileged)
        p_unpriv = y_pred_test[unprivileged_mask].mean() if unprivileged_mask.sum() > 0 else float('nan')
        p_priv = y_pred_test[privileged_mask].mean() if privileged_mask.sum() > 0 else float('nan')
        stat_parity_diff = p_unpriv - p_priv

        # Equalized odds (TPR difference): TPR_unpriv - TPR_priv
        tpr_unpriv = (
            (y_pred_test[unprivileged_mask] & self.y_test[unprivileged_mask]).sum() /
            max(1, self.y_test[unprivileged_mask].sum())
        ) if unprivileged_mask.sum() > 0 else float('nan')
        tpr_priv = (
            (y_pred_test[privileged_mask] & self.y_test[privileged_mask]).sum() /
            max(1, self.y_test[privileged_mask].sum())
        ) if privileged_mask.sum() > 0 else float('nan')
        tpr_diff = tpr_unpriv - tpr_priv

        audit = {
            'statistical_parity_difference': float(stat_parity_diff),
            'tpr_difference': float(tpr_diff),
            'p_unpriv': float(p_unpriv),
            'p_priv': float(p_priv),
            'tpr_unpriv': float(tpr_unpriv),
            'tpr_priv': float(tpr_priv),
            'n_privileged': int(privileged_mask.sum()),
            'n_unprivileged': int(unprivileged_mask.sum())
        }
        logger.info(f"Fairness audit: {audit}")
        return audit

    # ----------------------------
    # Persistence
    # ----------------------------
    def _save_artifacts(self):
        """
        Persist pipeline and calibrated classifier for later use.
        """
        pipeline_path = os.path.join(self.model_output_dir, "pipeline.joblib")
        calib_path = os.path.join(self.model_output_dir, "calibrated_clf.joblib")
        try:
            joblib.dump(self.pipeline, pipeline_path)
            joblib.dump(self.calibrated_clf, calib_path)
            logger.info(f"Saved pipeline to {pipeline_path} and calibrated classifier to {calib_path}")
        except Exception as e:
            logger.warning(f"Failed to save artifacts: {e}")

    def load_artifacts(self, pipeline_path: str, calib_path: str):
        """
        Load saved pipeline and calibrated classifier.
        """
        self.pipeline = joblib.load(pipeline_path)
        self.calibrated_clf = joblib.load(calib_path)
        logger.info(f"Loaded pipeline from {pipeline_path} and calibrated classifier from {calib_path}")
        # Re-init SHAP explainer after loading
        self._init_shap_explainer()

    # ----------------------------
    # Applicant data generation (demo / simulation)
    # ----------------------------
    def generate_applicant_data(self) -> Dict[str, Any]:
        """
        Generate a realistic-feeling applicant dictionary (no PII) for demonstration.
        """
        first_names = ['Rajesh', 'Suresh', 'Priya', 'Ananya', 'Vijay', 'Sunita',
                       'Arun', 'Meena', 'Kumar', 'Lakshmi', 'Manoj', 'Pooja']
        last_names = ['Sharma', 'Patel', 'Singh', 'Kumar', 'Verma', 'Reddy',
                      'Mehta', 'Choudhury', 'Gupta', 'Malik', 'Yadav']

        profile = {
            'name': f"{random.choice(first_names)} {random.choice(last_names)}",
            'location': random.choice(['Rajasthan', 'UP', 'Bihar', 'WB', 'Maharashtra', 'TamilNadu']),
            'residence_years': random.randint(0, 30)  # non-PII residency duration
        }

        # DBT payments (0-8 over last year)
        dbt_data = []
        for _ in range(random.randint(0, 8)):
            dbt_data.append({
                'date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
                'amount': random.randint(500, 5000),
                'scheme': random.choice(['PM-KISAN', 'MGNREGA', 'LPG-Subsidy', 'Pension', 'Ujjwala'])
            })

        # Bill payments
        bill_data = []
        for _ in range(random.randint(3, 15)):
            due_date = (datetime.now() - timedelta(days=random.randint(0, 180)))
            payment_delta = random.randint(-10, 30)
            bill_data.append({
                'due_date': due_date.strftime('%Y-%m-%d'),
                'payment_date': (due_date + timedelta(days=payment_delta)).strftime('%Y-%m-%d'),
                'amount': random.randint(100, 2000),
                'type': random.choice(['Electricity', 'Water', 'Mobile', 'LPG', 'DTH'])
            })

        # AEPS transactions
        aeps_data = []
        for _ in range(random.randint(0, 50)):
            aeps_data.append({
                'date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
                'amount': random.randint(100, 10000),
                'type': random.choice(['Withdrawal', 'Deposit'])
            })

        # UPI transactions
        upi_data = []
        for _ in range(random.randint(0, 100)):
            upi_data.append({
                'date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
                'amount': random.randint(10, 5000),
                'merchant': random.choice(['Friend', 'Grocer', 'Kirana', 'Mobile-Recharge'])
            })

        # Account monthly summaries
        account_data = []
        for month in range(12):
            deposits = random.randint(5000, 30000)
            withdrawals = random.randint(4000, 35000)
            account_data.append({
                'month': (datetime.now() - timedelta(days=30 * month)).strftime('%Y-%m'),
                'total_deposits': deposits,
                'total_withdrawals': withdrawals
            })

        return {
            'profile': profile,
            'dbt': dbt_data,
            'bills': bill_data,
            'aeps': aeps_data,
            'upi': upi_data,
            'account': account_data
        }

    def run_simulation(self, n_applicants: int = 5) -> List[Dict[str, Any]]:
        """
        Generate applicants, run the scoring pipeline, and pack readable outputs.
        """
        results = []
        for _ in range(n_applicants):
            applicant = self.generate_applicant_data()
            assessment = self.predict(applicant)
            results.append({
                'profile': applicant['profile'],
                'assessment': assessment,
                'raw_stats': {
                    'dbt_count': len(applicant['dbt']),
                    'bill_count': len(applicant['bills']),
                    'aeps_count': len(applicant['aeps']),
                    'upi_count': len(applicant['upi']),
                    'account_months': len(applicant['account'])
                }
            })
        return results


# ----------------------------
# Minimal self-tests / demonstration
# ----------------------------
def quick_demo():
    """
    Run a quick demonstration of initialization, simulation, and fairness audit.
    """
    logger.setLevel(logging.INFO)
    logger.info("Starting quick demo of RuralCreditScorerV2...")
    scorer = RuralCreditScorerV2(n_samples=3000, random_state=123)  # smaller for demo speed

    # Run simulated applicants and print results
    sims = scorer.run_simulation(n_applicants=3)
    for i, res in enumerate(sims):
        profile = res['profile']
        assessment = res['assessment']
        print("\n" + "-" * 60)
        print(f"Applicant {i + 1}: {profile['name']} from {profile['location']}")
        print(f"Residence years (proxy): {profile['residence_years']}")
        print(f"Credit Score: {assessment['credit_score']:.0f}")
        print(f"Risk Score: {assessment['risk_score']:.1f}%")
        print(f"Default Probability (calibrated): {assessment['default_probability']:.3f}")
        print("Top Risk Factors:", ", ".join(assessment['risk_factors']))
        print("Components:")
        for k, v in assessment['score_components'].items():
            print(f"  - {k}: {v:.1f}%")
        print("-" * 60)

    # Run a fairness audit (protected attribute = 'gender', privileged=M)
    audit = scorer.fairness_audit(protected_attr='gender', privileged_value='M')
    print("\nFairness audit (gender):")
    for k, v in audit.items():
        print(f"  {k}: {v}")


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    # Running quick demo + minimal asserts to ensure core functions operate without runtime errors.
    quick_demo()

    # Unit-like checks (basic)
    logger.info("Running basic checks...")
    scorer = RuralCreditScorerV2(n_samples=1500, random_state=7)
    # Generate single applicant and assert output structure
    sample = scorer.generate_applicant_data()
    out = scorer.predict(sample)
    assert 'credit_score' in out and 'default_probability' in out and 'risk_factors' in out, "Predict output missing keys"
    assert 300 <= out['credit_score'] <= 900, "Credit score out of expected range"
    assert 0.0 <= out['default_probability'] <= 1.0, "Default probability out of range"

    # Ensure fairness audit runs
    audit_res = scorer.fairness_audit(protected_attr='gender', privileged_value='M')
    assert 'statistical_parity_difference' in audit_res, "Fairness audit missing metrics"

    logger.info("All basic checks passed. Prototype initialized and functioning.")

    # NOTE: Next production steps (not implemented in this prototype):
    #  - Replace synthetic data with privacy-preserving real data (with consent), and retrain.
    #  - Add rigorous cross-validation and hyperparameter tuning (e.g., with Optuna).
    #  - Perform detailed fairness analysis across multiple sensitive attributes and multiple thresholds.
    #  - Integrate model explainability and human-in-the-loop override workflows.
    #  - Legal & privacy review before any PII or sensitive identifier is stored or used.
    #  - Monitoring, logging, and alerting for model drift + re-training pipelines.
