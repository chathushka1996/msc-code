#!/bin/bash
# Run all models for sl_piliyandala dataset
# Deep Learning: PatchTST, RNN, LSTM, GRU
# ML Baselines: Gradient Boosting, XGBoost, KNN, LightGBM, CatBoost

echo "==========================================="
echo "Running all models on sl_piliyandala dataset"
echo "==========================================="

# Run PatchTST
echo ""
echo ">>> Running PatchTST..."
bash patchtst.sh

# Run RNN
echo ""
echo ">>> Running RNN..."
bash rnn_sl_piliyandala.sh

# Run LSTM
echo ""
echo ">>> Running LSTM..."
bash lstm_sl_piliyandala.sh

# Run GRU
echo ""
echo ">>> Running GRU..."
bash gru_sl_piliyandala.sh

# Run ML Baselines (Gradient Boosting, XGBoost, KNN, LightGBM, CatBoost)
echo ""
echo ">>> Running ML Baselines..."
bash ml_baselines_sl_piliyandala.sh

echo ""
echo "==========================================="
echo "All model evaluations completed!"
echo "==========================================="
echo ""
echo "Results can be found in:"
echo "  - Deep Learning: ./drive/MyDrive/msc-val/logs/"
echo "  - ML Baselines: ./drive/MyDrive/msc-val/logs/ and ml_baseline_results.txt"
