version=urial.plus.best_4.examples
temp=0.5
rp=1
output_dir="result_dirs/mt-bench/urial_bench/"
mkdir -p $output_dir
gpu=0,1
n=2
CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --engine hf \
    --model_name mistral-community/Mixtral-8x22B-v0.1-4bit \
    --tensor_parallel_size $n \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 1 --max_tokens 2048 \
    --filepath $output_dir/Mixtral-8x22B-v0.1-4bit.urial.plus.best_4-hf/Mixtral-8x22B-v0.1-4bit.urial.plus.best_4-hf.turn1.json \
    --overwrite 


CUDA_VISIBLE_DEVICES=$gpu python src/unified_infer.py \
    --urial $version \
    --engine hf \
    --model_name mistral-community/Mixtral-8x22B-v0.1-4bit \
    --tensor_parallel_size $n \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/Mixtral-8x22B-v0.1-4bit.urial.plus.best_4-hf/Mixtral-8x22B-v0.1-4bit.urial.plus.best_4-hf.turn1.json \
    --top_p 1 --temperature $temp --repetition_penalty $rp --batch_size 1 --max_tokens 2048 \
    --filepath $output_dir/Mixtral-8x22B-v0.1-4bit.urial.plus.best_4-hf/Mixtral-8x22B-v0.1-4bit.urial.plus.best_4-hf.turn2.json \
    --overwrite 
