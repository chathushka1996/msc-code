"""
ML Baseline Models Evaluation for Time Series Forecasting
Evaluates: Gradient Boosting, XGBoost, KNN, LightGBM, CatBoost

Dataset: sl_piliyandala (Solar Power Output Forecasting)
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings
import time
import json

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed. Install with: pip install catboost")


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

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    return mae, mse, rmse, mape, mspe, rse, corr


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


def create_sequences(data, seq_len, pred_len, features='M'):
    """Create sequences for time series forecasting."""
    X, y = [], []
    n_features = data.shape[1]
    
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len].flatten())
        if features == 'M':
            y.append(data[i+seq_len:i+seq_len+pred_len].flatten())
        else:
            y.append(data[i+seq_len:i+seq_len+pred_len, -1])
    
    return np.array(X), np.array(y)


def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, 
                              pred_len, n_features, features='M'):
    """Train and evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    if y_train.ndim > 1 and y_train.shape[1] > 1:
        if not isinstance(model, MultiOutputRegressor):
            model = MultiOutputRegressor(model)
    
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    start_time = time.time()
    y_pred_test = model.predict(X_test)
    pred_time = time.time() - start_time
    
    if features == 'M':
        y_pred_test = y_pred_test.reshape(-1, pred_len, n_features)
        y_test_reshaped = y_test.reshape(-1, pred_len, n_features)
    else:
        y_pred_test = y_pred_test.reshape(-1, pred_len, 1)
        y_test_reshaped = y_test.reshape(-1, pred_len, 1)
    
    mae, mse, rmse, mape, mspe, rse, corr = metric(y_pred_test, y_test_reshaped)
    
    print(f"\nTest Results for {model_name}:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAPE: {mape:.6f}")
    print(f"  MSPE: {mspe:.6f}")
    print(f"  RSE:  {rse:.6f}")
    print(f"  CORR: {corr:.6f}")
    print(f"  Prediction time: {pred_time:.4f} seconds")
    
    return {
        'model': model_name,
        'mse': float(mse),
        'mae': float(mae),
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
    
    models['GradientBoosting'] = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=args.random_seed,
        verbose=0
    )
    
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=args.random_seed,
            verbosity=0,
            n_jobs=-1
        )
    
    models['KNN'] = KNeighborsRegressor(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    )
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=args.random_seed,
            verbose=-1,
            n_jobs=-1
        )
    
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostRegressor(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            random_seed=args.random_seed,
            verbose=0,
            thread_count=-1
        )
    
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
    
    args = parser.parse_args()
    
    np.random.seed(args.random_seed)
    
    os.makedirs(args.log_path, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ML Baseline Models Evaluation")
    print(f"{'='*60}")
    print(f"Dataset: {args.root_path}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Prediction Length: {args.pred_len}")
    print(f"Features: {args.features}")
    print(f"Target: {args.target}")
    print(f"{'='*60}\n")
    
    print("Loading data...")
    train_data, val_data, test_data, scaler, n_features = load_data(
        args.root_path, 
        target=args.target
    )
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Number of features: {n_features}")
    
    print("\nCreating sequences...")
    X_train, y_train = create_sequences(train_data, args.seq_len, args.pred_len, args.features)
    X_val, y_val = create_sequences(val_data, args.seq_len, args.pred_len, args.features)
    X_test, y_test = create_sequences(test_data, args.seq_len, args.pred_len, args.features)
    
    print(f"Training sequences: {X_train.shape[0]}")
    print(f"Validation sequences: {X_val.shape[0]}")
    print(f"Test sequences: {X_test.shape[0]}")
    print(f"Input features per sample: {X_train.shape[1]}")
    print(f"Output features per sample: {y_train.shape[1]}")
    
    all_models = get_models(args)
    
    if args.model.lower() != 'all':
        if args.model in all_models:
            models = {args.model: all_models[args.model]}
        else:
            print(f"Model {args.model} not found. Available: {list(all_models.keys())}")
            return
    else:
        models = all_models
    
    results = []
    
    for model_name, model in models.items():
        try:
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
                f.write(f"RMSE: {result['rmse']}\n")
                f.write(f"MAPE: {result['mape']}\n")
                f.write(f"MSPE: {result['mspe']}\n")
                f.write(f"RSE: {result['rse']}\n")
                f.write(f"CORR: {result['corr']}\n")
                f.write(f"Train Time: {result['train_time']}\n")
                f.write(f"Pred Time: {result['pred_time']}\n")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Summary of Results")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'MSE':>12} {'MAE':>12} {'RMSE':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} {r['mse']:>12.6f} {r['mae']:>12.6f} {r['rmse']:>12.6f}")
    
    with open("ml_baseline_results.txt", 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Dataset: {args.dataset}, Seq: {args.seq_len}, Pred: {args.pred_len}\n")
        f.write(f"{'='*60}\n")
        for r in results:
            f.write(f"{r['model']}: mse={r['mse']:.6f}, mae={r['mae']:.6f}, rmse={r['rmse']:.6f}\n")
    
    results_file = os.path.join(args.log_path, f"ml_results_{args.dataset}_{args.seq_len}_{args.pred_len}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
