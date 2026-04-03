# ML Baseline Models for sl_piliyandala dataset
# Models: Gradient Boosting, XGBoost, KNN, LightGBM, CatBoost
# Default: CPU (sklearn/XGB/LGBM/Cat). Add --use_ml_gpu for GPU tree libs if installed.
# Watch live: tail -f ./drive/MyDrive/msc-val/logs/ML_baselines_sl_piliyandala_96_96.log

path=./drive/MyDrive/msc-val
if [ ! -d "$path/logs" ]; then
    mkdir $path/logs -p
fi

seq_len=96
dataset=sl_piliyandala
root_path_name=./dataset/$dataset
random_seed=2021

# Run all ML baseline models for different prediction lengths
for pred_len in 96 192 336 720
do
    echo "Running ML baselines for pred_len=$pred_len"
    python -u run_ml_baselines.py \
        --random_seed $random_seed \
        --root_path $root_path_name \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --features M \
        --model all \
        --log_path $path/logs \
        --dataset $dataset \
        > $path/logs/ML_baselines_$dataset'_'$seq_len'_'$pred_len.log
done

echo "All ML baseline evaluations completed!"
