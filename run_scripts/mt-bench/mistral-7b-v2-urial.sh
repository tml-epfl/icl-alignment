version=urial
temp=0
rp=1.15
output_dir="result_dirs/mt-bench/urial_bench/"
mkdir -p $output_dir
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --model_name alpindale/Mistral-7B-v0.2-hf \
    --tensor_parallel_size 1 \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/Mistral-7B-v0.2.urial-vllm/Mistral-7B-v0.2.urial-vllm.turn1.json \
    --overwrite 


CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --model_name alpindale/Mistral-7B-v0.2-hf \
    --tensor_parallel_size 1 \
    --dtype bfloat16 \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/Mistral-7B-v0.2.urial-vllm/Mistral-7B-v0.2.urial-vllm.turn1.json \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 8 --max_tokens 2048 \
    --filepath $output_dir/Mistral-7B-v0.2.urial-vllm/Mistral-7B-v0.2.urial-vllm.turn2.json \
    --overwrite 