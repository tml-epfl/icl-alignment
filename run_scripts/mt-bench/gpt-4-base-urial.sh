version=urial
temp=0.2
rp=0
output_dir="result_dirs/mt-bench/urial_bench/"
mkdir -p $output_dir
engine="openai"
model="gpt-4-base"
api_key=""
api_org=""
python src/unified_infer_v2.py \
    --urial $version \
    --engine $engine \
    --model_name $model \
    --data_name mt-bench \
    --mt_turn 1 \
    --top_p 1 --temperature $temp --batch_size 1 --max_tokens 2048 --frequency_penalty 0.3 --presence_penalty 0.1 \
    --filepath $output_dir/gpt-4-base.urial/gpt-4-base.urial.turn1.json \
    --api_key $api_key \
    --api_org $api_org \
    --overwrite


python src/unified_infer_v2.py \
    --urial $version \
    --engine $engine \
    --model_name $model \
    --data_name mt-bench \
    --mt_turn 2 \
    --mt_turn1_result $output_dir/gpt-4-base.urial/gpt-4-base.urial.turn1.json \
    --top_p 1 --temperature $temp --batch_size 1 --max_tokens 2048 --frequency_penalty 0.3 --presence_penalty 0.1 \
    --filepath $output_dir/gpt-4-base.urial/gpt-4-base.urial.turn2.json \
    --overwrite \
    --api_key $api_key \
    --api_org $api_org