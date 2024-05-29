version=urial.plus.best_4.examples
rp=1
N=1
T=0.5
output_dir="result_dirs/alpaca_eval"
mkdir -p $output_dir
gpu=0,1
tps=2
model_name="mistral-community/Mixtral-8x22B-v0.1-4bit"
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --engine hf \
    --model_name $model_name \
    --tensor_parallel_size $tps \
    --data_name alpaca_eval --num_outputs $N \
    --top_p 1.0 --temperature $T --repetition_penalty $rp  --batch_size 16 --max_tokens 2048 \
    --filepath $output_dir/Mixtral-8x22B-v0.1-4bit-urial.plus.best.4.examples.json \
    --overwrite