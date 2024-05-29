version=urial.plus.best_4.examples
rp=1.15
N=1
T=0
output_dir="result_dirs/alpaca_eval"
mkdir -p $output_dir
gpu=0
tps=1
model_name="alpindale/Mistral-7B-v0.2-hf"
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --model_name $model_name \
    --tensor_parallel_size $tps \
    --dtype bfloat16 \
    --data_name alpaca_eval --num_outputs $N \
    --top_p 1.0 --temperature $T --repetition_penalty $rp  --batch_size 16 --max_tokens 2048 \
    --filepath $output_dir/Mistral-7B-v0.2-urial.plus.best_4.json \
    --overwrite