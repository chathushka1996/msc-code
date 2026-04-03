"""
ML Baseline Models Evaluation for Time Series Forecasting
Evaluates: Gradient Boosting, XGBoost, KNN, LightGBM, CatBoost

Dataset: sl_piliyandala (Solar Power Output Forecasting)

Hardware: This script does NOT use PyTorch/CUDA. Tree/KNN models run on CPU
(multi-threaded where supported). Use run_longExp.py for GPU deep learning.
Optional: --use_ml_gpu enables GPU for XGBoost/LightGBM/CatBoost when available.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone
import warnings
import time
import json

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

warnings.filterwarnings('ignore')


def _print(msg="", **kwargs):
    print(msg, flush=True, **kwargs)


def print_runtime_environment(use_ml_gpu=False):
    """Clarify CPU vs GPU so long runs do not look 'stuck' with no context."""
    _print("\n" + "=" * 60)
    _print("RUNTIME / DEVICE")
    _print("=" * 60)
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            _print(f"  PyTorch CUDA available: yes (device: {torch.cuda.get_device_name(0)})")
        else:
            _print("  PyTorch CUDA available: no")
    except Exception as e:
        _print(f"  PyTorch: not checked ({e})")
    _print("  This script (ML baselines): uses sklearn / XGBoost / LightGBM / CatBoost — not PyTorch.")
    _print("  Default backend: CPU (n_jobs=-1 where supported).")
    if use_ml_gpu:
        _print("  --use_ml_gpu: will try GPU for XGBoost/LightGBM/CatBoost if drivers/libs allow.")
    else:
        _print("  GPU for tree libs: off (pass --use_ml_gpu to try XGB/LGBM/Cat GPU).")
    _print("=" * 60 + "\n")


def fit_multioutput_with_progress(base_estimator, X, y, model_name, use_tqdm=True):
    """
    Fit one estimator per output column with progress. MultiOutputRegressor can
    train hundreds of sub-models (seq_len * n_features outputs) with no logs — this fixes that.
    """
    n_out = y.shape[1]
    estimators = [clone(base_estimator) for _ in range(n_out)]
    iterator = range(n_out)
    if use_tqdm and tqdm is not None:
        iterator = tqdm(
            iterator,
            desc=f"{model_name} ({n_out} outputs)",
            file=sys.stdout,
            mininterval=1.0,
            unit="out",
        )
    elif tqdm is None and n_out > 50:
        _print(f"  (install tqdm for a progress bar: pip install tqdm)")

    for i in iterator:
        if tqdm is None and n_out > 20 and (i % max(1, n_out // 10) == 0 or i == n_out - 1):
            _print(f"  [{model_name}] training output dimension {i + 1}/{n_out} ...")
        estimators[i].fit(X, y[:, i])

    mor = MultiOutputRegressor(base_estimator)
    mor.estimators_ = estimators
    mor.n_outputs_ = n_out
    return mor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    _print("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    _print("LightGBM not installed. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    _print("CatBoost not installed. Install with: pip install catboost")


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / (true + 1e-8)))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + 1e-8)))

def R2(pred, true):
    """R-squared (coefficient of determination)"""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    r2 = R2(pred, true)
    return mae, mse, rmse, mape, mspe, rse, corr, r2


def load_data(root_path, target='Solar Power Output'):
    """Load and preprocess the dataset."""
    train_df = pd.read_csv(os.path.join(root_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(root_path, "val.csv"))
    test_df = pd.read_csv(os.path.join(root_path, "test.csv"))
    
    cols = list(train_df.columns)
    cols.remove(target)
    cols.remove('date')
    feature_cols = cols + [target]
    
    train_data = train_df[feature_cols].values
    val_data = val_df[feature_cols].values
    test_data = test_df[feature_cols].values
    
    scaler = StandardScaler()
    scaler.fit(train_data)
    
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    return train_scaled, val_scaled, test_scaled, scaler, len(feature_cols)


def create_sequences(data, seq_len, pred_len, features='M', desc="Building sequences"):
    """Create sequences for time series forecasting."""
    X, y = [], []
    n = len(data) - seq_len - pred_len + 1
    n_features = data.shape[1]
    iterator = range(n)
    if tqdm is not None and n > 5000:
        iterator = tqdm(iterator, desc=desc, file=sys.stdout, mininterval=1.0, unit="win")

    for i in iterator:
        X.append(data[i:i + seq_len].flatten())
        if features == 'M':
            y.append(data[i + seq_len:i + seq_len + pred_len].flatten())
        else:
            y.append(data[i + seq_len:i + seq_len + pred_len, -1])

    return np.array(X), np.array(y)


def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, 
                              pred_len, n_features, features='M'):
    """Train and evaluate a single model."""
    _print(f"\n{'='*60}")
    _print(f"Training {model_name} ...")
    _print(f"  (y has {y_train.shape[1]} output dimensions — each is fitted separately; this can take long on CPU)")
    _print(f"{'='*60}")
    
    start_time = time.time()
    
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        if not isinstance(model, MultiOutputRegressor):
            model = fit_multioutput_with_progress(
                model, X_train, y_train, model_name, use_tqdm=(tqdm is not None)
            )
        else:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - start_time
    _print(f"Training completed in {train_time:.2f}s")
    
    _print(f"Running inference on test set (n={X_test.shape[0]}) ...")
    start_time = time.time()
    y_pred_test = model.predict(X_test)
    pred_time = time.time() - start_time
    _print(f"Inference done in {pred_time:.2f}s")
    
    if features == 'M':
        y_pred_test = y_pred_test.reshape(-1, pred_len, n_features)
        y_test_reshaped = y_test.reshape(-1, pred_len, n_features)
    else:
        y_pred_test = y_pred_test.reshape(-1, pred_len, 1)
        y_test_reshaped = y_test.reshape(-1, pred_len, 1)
    
    mae, mse, rmse, mape, mspe, rse, corr, r2 = metric(y_pred_test, y_test_reshaped)
    
    _print(f"\nTest Results for {model_name}:")
    _print(f"  MSE:  {mse:.6f}")
    _print(f"  MAE:  {mae:.6f}")
    _print(f"  R²:   {r2:.6f}")
    _print(f"  RMSE: {rmse:.6f}")
    _print(f"  MAPE: {mape:.6f}")
    _print(f"  MSPE: {mspe:.6f}")
    _print(f"  RSE:  {rse:.6f}")
    _print(f"  CORR: {corr:.6f}")
    _print(f"  Prediction time: {pred_time:.4f}s")
    
    return {
        'model': model_name,
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'rmse': float(rmse),
        'mape': float(mape),
        'mspe': float(mspe),
        'rse': float(rse),
        'corr': float(corr),
        'train_time': train_time,
        'pred_time': pred_time
    }


def get_models(args):
    """Get dictionary of models to evaluate."""
    models = {}
    gpu = getattr(args, "use_ml_gpu", False)

    models['GradientBoosting'] = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=args.random_seed,
        verbose=0,
    )

    if HAS_XGBOOST:
        xgb_kw = dict(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=args.random_seed,
            verbosity=0,
            n_jobs=-1,
        )
        if gpu:
            xgb_kw["tree_method"] = "gpu_hist"
            xgb_kw["predictor"] = "gpu_predictor"
        else:
            xgb_kw["tree_method"] = "hist"
        models['XGBoost'] = xgb.XGBRegressor(**xgb_kw)

    models['KNN'] = KNeighborsRegressor(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1,
    )

    if HAS_LIGHTGBM:
        lgb_kw = dict(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=args.random_seed,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_kw["device"] = "gpu" if gpu else "cpu"
        models['LightGBM'] = lgb.LGBMRegressor(**lgb_kw)

    if HAS_CATBOOST:
        cb_kw = dict(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_seed=args.random_seed,
            verbose=0,
            thread_count=-1,
        )
        cb_kw["task_type"] = "GPU" if gpu else "CPU"
        models['CatBoost'] = CatBoostRegressor(**cb_kw)

    return models


def main():
    parser = argparse.ArgumentParser(description='ML Baseline Models for Time Series Forecasting')
    
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
    parser.add_argument('--root_path', type=str, default='./dataset/sl_piliyandala', help='root path of the data')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'],
                        help='M:multivariate, S:univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='Solar Power Output', help='target feature')
    parser.add_argument('--model', type=str, default='all', 
                        help='model name: GradientBoosting, XGBoost, KNN, LightGBM, CatBoost, or all')
    parser.add_argument('--log_path', type=str, default='./logs', help='path to save logs')
    parser.add_argument('--dataset', type=str, default='sl_piliyandala', help='dataset name for logging')
    parser.add_argument(
        '--use_ml_gpu',
        action='store_true',
        help='Use GPU for XGBoost/LightGBM/CatBoost if available (GradientBoosting/KNN stay CPU)',
    )

    args = parser.parse_args()

    np.random.seed(args.random_seed)

    os.makedirs(args.log_path, exist_ok=True)

    print_runtime_environment(use_ml_gpu=args.use_ml_gpu)

    _print(f"\n{'='*60}")
    _print("ML Baseline Models Evaluation")
    _print(f"{'='*60}")
    _print(f"Dataset: {args.root_path}")
    _print(f"Sequence Length: {args.seq_len}")
    _print(f"Prediction Length: {args.pred_len}")
    _print(f"Features: {args.features}")
    _print(f"Target: {args.target}")
    _print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _print(f"{'='*60}\n")

    t0 = time.time()
    _print("[1/4] Loading CSVs ...")
    train_data, val_data, test_data, scaler, n_features = load_data(
        args.root_path, 
        target=args.target
    )
    _print(f"  Train rows: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)} | n_features: {n_features}")
    _print(f"  Loaded in {time.time() - t0:.1f}s\n")

    _print("[2/4] Building sliding windows (this can take ~1–3 min on large CSVs) ...")
    t1 = time.time()
    X_train, y_train = create_sequences(
        train_data, args.seq_len, args.pred_len, args.features, desc="train windows"
    )
    X_val, y_val = create_sequences(
        val_data, args.seq_len, args.pred_len, args.features, desc="val windows"
    )
    X_test, y_test = create_sequences(
        test_data, args.seq_len, args.pred_len, args.features, desc="test windows"
    )
    _print(f"  Windows built in {time.time() - t1:.1f}s")
    _print(f"  Train windows: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    _print(f"  X dim: {X_train.shape[1]} | y dim (outputs): {y_train.shape[1]}")
    _print(f"  Note: each baseline fits one sub-model per output dim — expect long CPU time without GPU.\n")

    _print("[3/4] Training models (see per-model progress below) ...")
    
    all_models = get_models(args)
    
    if args.model.lower() != 'all':
        if args.model in all_models:
            models = {args.model: all_models[args.model]}
        else:
            _print(f"Model {args.model} not found. Available: {list(all_models.keys())}")
            return
    else:
        models = all_models

    n_models = len(models)
    results = []

    for mi, (model_name, model) in enumerate(models.items(), start=1):
        try:
            _print(f"\n>>> [{mi}/{n_models}] Starting: {model_name} @ {time.strftime('%H:%M:%S')}")
            result = train_and_evaluate_model(
                model, model_name,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                args.pred_len, n_features,
                args.features
            )
            result['seq_len'] = args.seq_len
            result['pred_len'] = args.pred_len
            results.append(result)
            
            log_file = os.path.join(args.log_path, f"{model_name}_{args.dataset}_{args.seq_len}_{args.pred_len}.log")
            with open(log_file, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Dataset: {args.dataset}\n")
                f.write(f"Seq Length: {args.seq_len}\n")
                f.write(f"Pred Length: {args.pred_len}\n")
                f.write(f"MSE: {result['mse']}\n")
                f.write(f"MAE: {result['mae']}\n")
                f.write(f"R2: {result['r2']}\n")
                f.write(f"RMSE: {result['rmse']}\n")
                f.write(f"MAPE: {result['mape']}\n")
                f.write(f"MSPE: {result['mspe']}\n")
                f.write(f"RSE: {result['rse']}\n")
                f.write(f"CORR: {result['corr']}\n")
                f.write(f"Train Time: {result['train_time']}\n")
                f.write(f"Pred Time: {result['pred_time']}\n")
            
        except Exception as e:
            _print(f"Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    _print(f"\n[4/4] Summary @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _print(f"\n{'='*80}")
    _print("Summary of Results")
    _print(f"{'='*80}")
    _print(f"{'Model':<20} {'MSE':>12} {'MAE':>12} {'R²':>12} {'RMSE':>12}")
    _print("-" * 80)
    for r in results:
        _print(f"{r['model']:<20} {r['mse']:>12.6f} {r['mae']:>12.6f} {r['r2']:>12.6f} {r['rmse']:>12.6f}")
    
    with open("ml_baseline_results.txt", 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Dataset: {args.dataset}, Seq: {args.seq_len}, Pred: {args.pred_len}\n")
        f.write(f"{'='*80}\n")
        for r in results:
            f.write(f"{r['model']}: mse={r['mse']:.6f}, mae={r['mae']:.6f}, r2={r['r2']:.6f}, rmse={r['rmse']:.6f}\n")
    
    results_file = os.path.join(args.log_path, f"ml_results_{args.dataset}_{args.seq_len}_{args.pred_len}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    _print(f"\nDone. Results saved to {results_file}")


if __name__ == '__main__':
    main()
