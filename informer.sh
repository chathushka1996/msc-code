path=.
if [ ! -d "$path/logs" ]; then
    mkdir $path/logs
fi

if [ ! -d "$path/logs/LongForecasting" ]; then
    mkdir $path/logs/LongForecasting
fi

seq_len=96
dataset=sl
root_path_name=./data/$dataset
data_path_name=solar.csv
model_id_name=solar_$dataset
data_name=custom
random_seed=2021
checkpoints=$path/model_log
model_name=Informer

for pred_len in 96
do
  python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 20\
    --patience 5\
    --checkpoints $checkpoints\
    --train_epochs 20 > $path/logs/LongForecasting/$model_name'_'$model_id_name'_'$pred_len.log
done