export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer
seq_len=96
dataset=sl_t
root_path_name=./data/$dataset
data_path_name=solar.csv
model_id_name=solar_$dataset
data_name=custom
pred_len=96
random_seed=2021
checkpoints=$path/model_log
for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --d_model 512\
    --d_ff 512\
    --itr 1 \
    --train_epochs 100\
    --patience 5\
    --checkpoints $checkpoints > $path/logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done