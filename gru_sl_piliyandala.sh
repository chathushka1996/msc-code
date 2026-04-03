# GRU model for sl_piliyandala dataset
path=./drive/MyDrive/msc-val
if [ ! -d "$path/logs" ]; then
    mkdir $path/logs -p
fi

seq_len=96
model_name=GRU
dataset=sl_piliyandala
root_path_name=./dataset/$dataset
data_path_name=solar.csv
model_id_name=solar_$dataset
data_name=custom
pred_len=96
random_seed=2021
checkpoints=$path/models/

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.2\
        --des 'Exp' \
        --train_epochs 50\
        --patience 5\
        --checkpoints $checkpoints\
        --itr 1 --batch_size 128 --learning_rate 0.0001 > $path/logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
