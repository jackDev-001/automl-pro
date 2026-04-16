"""
automl_engine.py  ──  AutoML Pro v2.0
──────────────────────────────────────
Features:
  • Smart preprocessing  (missing values, outliers, scaling, encoding)
  • Auto problem detection (regression / classification)
  • 6+ model ensemble with CV scoring
  • Hyperparameter tuning (RandomForest, GBM, XGBoost if available)
  • Feature selection via variance + importance
  • Detailed logging & training-time tracking
"""

import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, r2_score, f1_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold

# ── Classifiers ──────────────────────────────
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Returns (X_processed, y, meta_dict)
    meta_dict contains encoders / scalers so they can be reused at inference time.
    """
    df = df.copy()
    meta = {"label_encoders": {}, "scaler": None, "dropped_cols": [], "target_encoder": None}

    # ── 1a. Separate target ──────────────────
    y_raw = df[target].copy()
    df = df.drop(columns=[target])

    # ── 1b. Drop near-constant & high-cardinality cols ──
    for col in df.columns:
        if df[col].nunique() <= 1:
            df.drop(columns=[col], inplace=True)
            meta["dropped_cols"].append(col)
        elif df[col].dtype == "object" and df[col].nunique() > 50:
            df.drop(columns=[col], inplace=True)
            meta["dropped_cols"].append(col)

    # ── 1c. Missing values ───────────────────
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
            else:
                # Use median (robust to outliers vs mean)
                df[col].fillna(df[col].median(), inplace=True)

    # ── 1d. Encode categoricals ──────────────
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            meta["label_encoders"][col] = le

    # ── 1e. Clip extreme outliers (IQR × 3) ─
    for col in df.select_dtypes(include=np.number).columns:
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q1, upper=q3)

    # ── 1f. Remove near-zero variance ────────
    vt = VarianceThreshold(threshold=0.01)
    X_vt = vt.fit_transform(df)
    kept_cols = df.columns[vt.get_support()].tolist()
    df = pd.DataFrame(X_vt, columns=kept_cols)

    # ── 1g. Scale features (RobustScaler handles outliers better) ──
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df)
    df = pd.DataFrame(X_scaled, columns=df.columns)
    meta["scaler"] = scaler

    # ── 1h. Encode target if categorical ─────
    y = y_raw.reset_index(drop=True)
    if y.dtype == "object":
        le_y = LabelEncoder()
        y = pd.Series(le_y.fit_transform(y.astype(str)), name=target)
        meta["target_encoder"] = le_y
    else:
        y = y.reset_index(drop=True)
        y.fillna(y.median(), inplace=True)

    print(f"[AutoML] Preprocessed → X: {df.shape}  |  Missing: {df.isnull().sum().sum()}")
    return df, y, meta


# ─────────────────────────────────────────────
# 2. PROBLEM DETECTION
# ─────────────────────────────────────────────
def detect_problem(user_input: str, y: pd.Series) -> str:
    text = user_input.lower()

    # Hard keyword signals
    regression_words  = ["price", "cost", "revenue", "salary", "amount", "sales",
                         "predict", "forecast", "estimate", "value", "rate", "score"]
    classification_words = ["classify", "class", "category", "label", "detect",
                            "spam", "fraud", "churn", "sentiment", "type", "group"]

    reg_hits  = sum(w in text for w in regression_words)
    cls_hits  = sum(w in text for w in classification_words)

    # Also look at target cardinality & dtype
    n_unique = y.nunique()
    is_float_target = y.dtype in [np.float64, np.float32]

    if cls_hits > reg_hits:
        return "classification"
    if reg_hits > cls_hits:
        return "regression"
    # Fallback: use data shape
    if is_float_target or n_unique > 20:
        return "regression"
    return "classification"


# ─────────────────────────────────────────────
# 3. MODEL REGISTRY
# ─────────────────────────────────────────────
def get_models(problem: str) -> dict:
    if problem == "classification":
        models = {
            "LogisticRegression":    LogisticRegression(max_iter=1000, C=1.0),
            "DecisionTree":          DecisionTreeClassifier(max_depth=8, random_state=42),
            "RandomForest":          RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "ExtraTrees":            ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting":      GradientBoostingClassifier(n_estimators=100, random_state=42),
            "KNN":                   KNeighborsClassifier(n_neighbors=5),
            "NaiveBayes":            GaussianNB(),
        }
        # Try to add SVC (can be slow on large data)
        models["SVC"] = SVC(probability=True, kernel="rbf", C=1.0)

    else:
        models = {
            "Ridge":                 Ridge(alpha=1.0),
            "DecisionTree":          DecisionTreeRegressor(max_depth=8, random_state=42),
            "RandomForest":          RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "ExtraTrees":            ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting":      GradientBoostingRegressor(n_estimators=100, random_state=42),
            "KNN":                   KNeighborsRegressor(n_neighbors=5),
            "SVR":                   SVR(kernel="rbf", C=1.0),
        }

    # XGBoost (optional — only if installed)
 
# ─────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
TUNE_PARAMS = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth":    [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "ExtraTrees": {
        "n_estimators": [100, 200],
        "max_depth":    [None, 10],
    },
    "GradientBoosting": {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 5],
    },
    "XGBoost": {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [4, 6],
    },
}

def tune_model(name: str, model, X: pd.DataFrame, y: pd.Series, cv) -> object:
    if name not in TUNE_PARAMS:
        return model
    print(f"[AutoML] Tuning {name}...")
    grid = GridSearchCV(model, TUNE_PARAMS[name], cv=cv, n_jobs=-1, scoring="accuracy" if hasattr(y, "nunique") and y.nunique() <= 20 else "r2")
    grid.fit(X, y)
    print(f"[AutoML] {name} best params: {grid.best_params_}")
    return grid.best_estimator_


# ─────────────────────────────────────────────
# 5. SCORING HELPER
# ─────────────────────────────────────────────
def score_model(model, X: pd.DataFrame, y: pd.Series, problem: str, cv) -> float:
    """
    Classification → weighted F1 (handles class imbalance better than accuracy)
    Regression     → R²
    """
    if problem == "classification":
        scoring = "f1_weighted"
    else:
        scoring = "r2"

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return float(scores.mean())


# ─────────────────────────────────────────────
# 6. FEATURE SELECTION (post-training)
# ─────────────────────────────────────────────
def select_top_features(model, X: pd.DataFrame, top_n: int = 20) -> list[str]:
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=X.columns)
        return fi.nlargest(top_n).index.tolist()
    return X.columns.tolist()


# ─────────────────────────────────────────────
# 7. MAIN auto_ml FUNCTION
# ─────────────────────────────────────────────
def auto_ml(df: pd.DataFrame, target: str, user_input: str,
            tune_top_n: int = 2, cv_folds: int = 5) -> tuple:
    """
    Parameters
    ----------
    df          : raw DataFrame (as uploaded by user)
    target      : name of the target column
    user_input  : free-text problem description from user
    tune_top_n  : how many top models to hyper-tune (default 2)
    cv_folds    : cross-validation folds (default 5)

    Returns
    -------
    best_model, best_score, all_scores_dict, problem_type, X_processed
    """
    total_start = time.time()
    print("\n" + "="*55)
    print("  AutoML Pro v2.0  —  Training Started")
    print("="*55)

    # ── Step 1: Preprocess ───────────────────
    X, y, meta = preprocess(df, target)
    n_samples = len(X)

    # ── Step 2: Problem Detection ────────────
    problem = detect_problem(user_input, y)
    print(f"[AutoML] Problem type detected: {problem.upper()}")

    # ── Step 3: Choose CV strategy ───────────
    if problem == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # ── Step 4: Reduce CV for large datasets ─
    if n_samples > 10_000:
        print(f"[AutoML] Large dataset ({n_samples} rows) — using 3-fold CV to save time.")
        cv_folds = 3
        if problem == "classification":
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # ── Step 5: Get model registry ───────────
    models = get_models(problem)
    print(f"[AutoML] {len(models)} models in ensemble: {list(models.keys())}")

    # ── Step 6: First-pass scoring ───────────
    all_scores = {}
    print("\n[AutoML] First-pass cross-validation...")

    for name, model in models.items():
        t0 = time.time()
        try:
            sc = score_model(model, X, y, problem, cv)
            all_scores[name] = round(sc, 4)
            print(f"  {name:<25} score={sc:.4f}  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  {name:<25} FAILED: {e}")
            all_scores[name] = 0.0

    # ── Step 7: Pick top-N to tune ───────────
    sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    top_names = [n for n, _ in sorted_models[:tune_top_n] if n in TUNE_PARAMS]

    print(f"\n[AutoML] Tuning top models: {top_names}")
    for name in top_names:
        model = models[name]
        tuned = tune_model(name, model, X, y, cv)
        models[name] = tuned
        sc = score_model(tuned, X, y, problem, cv)
        all_scores[name] = round(sc, 4)
        print(f"  {name} after tuning → {sc:.4f}")

    # ── Step 8: Select best ──────────────────
    best_name = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_name]
    best_model = models[best_name]

    print(f"\n[AutoML] 🏆 Best model: {best_name}  |  Score: {best_score:.4f}")

    # ── Step 9: Final fit on full data ───────
    best_model.fit(X, y)
    print(f"[AutoML] Final model fitted on {n_samples} samples.")

    # ── Step 10: Summary ─────────────────────
    elapsed = time.time() - total_start
    print(f"\n[AutoML] ✅ Done in {elapsed:.1f}s")
    print("="*55 + "\n")

    return best_model, best_score, all_scores, problem, X