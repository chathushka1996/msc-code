# add --individual for DLinear-I
path=./drive/MyDrive/msc-val
if [ ! -d "$path/logs" ]; then
    mkdir $path/logs -p
fi

seq_len=96
model_name=NLinear
dataset=sl_t
root_path_name=./data/$dataset
data_path_name=solar.csv
model_id_name=solar_$dataset
data_name=custom
random_seed=2021
checkpoints=$path/models/
for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --train_epochs 20\
    --patience 5\
    --checkpoints $checkpoints\
    --itr 1 --batch_size 16  > $path/logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done