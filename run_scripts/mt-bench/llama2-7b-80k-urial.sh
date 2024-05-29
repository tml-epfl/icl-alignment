version=urial.plus.best_4.examples
temp=0
rp=1.15
output_dir="result_dirs/mt-bench/urial_bench/"
mkdir -p $output_dir
gpu=0
tsp=1
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --model_name yaofu/llama-2-7b-80k \
    --tensor_parallel_size ${tsp} \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/Llama-2-7b-80k.urial.plus.best_4-vllm/Llama-2-7b-80k.urial.plus.best_4-vllm.turn1.json \
    --overwrite 


CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --model_name yaofu/llama-2-7b-80k \
    --tensor_parallel_size ${tsp} \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/Llama-2-7b-80k.urial.plus.best_4-vllm/Llama-2-7b-80k.urial.plus.best_4-vllm.turn1.json \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/Llama-2-7b-80k.urial.plus.best_4-vllm/Llama-2-7b-80k.urial.plus.best_4-vllm.turn2.json \
    --overwrite
    