#!/bin/bash


script_dir=$(dirname "$(readlink -f "$0")")


output_string=$(python3 "$script_dir/string_match.py" | grep -E 'Product Type|Species')
output_t5l=$(python3 "$script_dir/t5.py" --model large | grep -E 'Product Type|Species')
output_t5xl=$(python3 "$script_dir/t5.py" --model xl | grep -E 'Product Type|Species')
output_t5llama=$(python3 "$script_dir/mistral.py" --finetune yes | grep -E 'Product Type|Species')
output_t5mistral=$(python3 "$script_dir/llama.py" --finetune yes | grep -E 'Product Type|Species')



echo "Results for string match"
echo "$output_string"
echo -e "\n"
echo "Results for t5 large"
echo "$output_t5l"
echo -e "\n"
echo "Results for t5 xl"
echo "$output_t5xl"
echo -e "\n"
echo "Results for t5 large"
echo "$output_t5llama"
echo -e "\n"
echo "Results for t5 large"
echo "$output_t5mistral"
echo -e "\n"