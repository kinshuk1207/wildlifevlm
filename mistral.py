from huggingface_hub import login

from huggingface_hub import snapshot_download

snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.2", local_dir="/tmp/mistral", local_dir_use_symlinks=False, cache_dir="/tmp")



from datasets import load_dataset
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Llama2 Model Training Script")
parser.add_argument("--finetune", type=str, default="no", choices=["yes", "no"],
                    help="To finetune or not.")

targs = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

def generate_jsonl_files_for_mistral(title_file, description_file, ground_file, train_file, val_file, split_ratio=0.8):
    # Read the data files
    with open(title_file, 'r') as f:
        titles = f.read().strip().split('\n')
    with open(description_file, 'r') as f:
        descriptions = f.read().strip().split('\n')
    with open(ground_file, 'r') as f:
        grounds = f.read().strip().split('Sample_')[1:]  # Ignore the first empty entry

    # Prepare the dataset
    data = []
    for title, description, ground in zip(titles, descriptions, grounds):
        product_type = ground.split('\n')[1].replace('Product Type: ', '')
        species = ground.split('\n')[2].replace('Species: ', '')

        prompt = f"[INST]\nGiven an ad listing with the title '{title}' and description '{description}', answer the following questions without any explanation or extra text.\nIdentify the species mentioned in the text, including specific names, e.g., 'Nile crocodile' instead of just 'crocodile'.\nSelect the product type from the following options: Animal fibers, Animal parts (bone or bone-like), Animal parts (fleshy), Coral product, Egg, Extract, Food, Ivory products, Live, Medicine, Nests, Organs and tissues, Powder, Scales or spines, Shells, Skin or leather products, Taxidermy, Insects.\nThe response should be in the format: Product Type:  Species: [/INST]"
        response = f"Product Type: {product_type}\n Species: {species}"
        full_text = f"<s>{prompt} \n{response}</s>"
        data.append({"text": full_text})

    # Split the data into training and validation
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    # Save to jsonl files
    with open(train_file, 'w') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
    with open(val_file, 'w') as f:
        for entry in val_data:
            f.write(json.dumps(entry) + '\n')

    return 'JSONL files created successfully!'


generate_jsonl_files_for_mistral(os.path.join(script_dir, 'data', 'title_finetune.txt'), os.path.join(script_dir, 'data', 'description_finetune.txt'), os.path.join(script_dir, 'data', 'ground_finetune.txt'), os.path.join(script_dir, 'tmp', 'finetunemistraltrain.jsonl'), os.path.join(script_dir, 'tmp', 'finetunemistraltest.jsonl'))



data_files = {
    "train": os.path.join(script_dir, 'tmp', 'finetunemistraltrain.jsonl'),
    "test": os.path.join(script_dir, 'tmp', 'finetunemistraltest.jsonl')
}

dataset = load_dataset("json", data_files=data_files)

import os 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from itertools import islice
from bs4 import BeautifulSoup
# from speciescitessearch import find_matches_and_species_info
import numpy as np
from tqdm import tqdm

seed_run = 66
torch.manual_seed(seed_run)
np.random.seed(seed_run)

modelpath="/tmp/mistral"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(modelpath, 
                                             quantization_config=bnb_config,
                                            #  attn_implementation="flash_attention_2"
                                             )

# Load (slow) Tokenizer, fast tokenizer sometimes ignores added tokens
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)   

# Add tokens <|im_start|> and <|im_end|>, latter is special eos token 
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.unk_token
tokenizer.add_eos_token = False
tokenizer.add_bos_token = False
# model.resize_token_embeddings(len(tokenizer))
# model.config.eos_token_id = tokenizer.eos_token_id

def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    )

dataset_tokenized = dataset.map(
    tokenize, 
    batched=True, 
    num_proc=os.cpu_count(),    # multithreaded
    remove_columns=["text"]     # don't need this anymore, we have tokens from here on
)

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

# from accelerate import FullyShardedDataParallelPlugin, Accelerator
# from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

# fsdp_plugin = FullyShardedDataParallelPlugin(
#     state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
#     optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),r
# )

# accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
# model = accelerator.prepare_model(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True
    print('more than one GPU detected')

import transformers
from datetime import datetime

project = "wildlife-finetune"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "/tmp/" + run_name


bs=2      # batch size
ga_steps=1  # gradient acc. steps
epochs=5
steps_per_epoch=len(dataset_tokenized["train"])//(bs*ga_steps)

trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=2,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga_steps,
        num_train_epochs=3,
        # max_steps=100,
        learning_rate=2.5e-4, # Want a small lr for finetuning
        # bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=steps_per_epoch,              # When to start reporting loss
        logging_dir="/tmp/logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=steps_per_epoch,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=steps_per_epoch,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        # report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. re-enable for inference

if targs.finetune == 'yes':
    trainer.train()


system_prompt = "The prompt will have an advertisement with the image description and the title of the ad. You need to answer the following questions about the ad. WildLife Product: Yes/No \nProduct Type: Ivory Products/Taxidermy/Live Animals/Animal Skins/Animal Fur/Bones and Skeletons/Horns and Antlers/Shells/Teeth and Claws/Traditional Medicines/Feathers/Coral and Sea Products/Eggs/Aquatic Life/Fossilized Remains/Insects/Leather Goods \nSpecies: \nTrade Status: (Not in CITES/CITES Appendix I/CITES Appendix II/CITES Appendix III) \nConservation Status: (Extinct (EX)/Extinct in the Wild (EW)/Critically Endangered (CR)/Endangered (EN)/Vulnerable (VU)/Near Threatened (NT)/Least Concern (LC)/Data Deficient (DD)/Not Evaluated (NE))."

# Read titles and descriptions from text files
with open(os.path.join(script_dir, 'data', 'title_infer.txt'), 'r') as f:
    titles = f.readlines()
# titles = titles[:10]
with open(os.path.join(script_dir, 'data', 'description_infer.txt'), 'r') as f:
    descriptions = f.readlines()
# descriptions = descriptions[:10]
product_types = [
    "Animal fibers", "Animal parts (bone or bone-like)", "Animal parts (fleshy)",
    "Coral product", "Egg", "Extract", "Food", "Ivory products", "Live", "Medicine",
    "Nests", "Organs and tissues", "Powder", "Scales or spines", "Shells",
    "Skin or leather products", "Taxidermy", "Insects"
]

def check_product_type(generated_text):
    """
    Check if the generated product type is in the predefined list of options.
    Returns True if a valid product type is found, False otherwise.
    """
    for pt in product_types:
        if pt in generated_text:
            return True  # Valid product type found
    return False  # No valid product type found

# Ensure both files have the same number of lines
assert len(titles) == len(descriptions), "Titles and descriptions files must have the same number of lines."
gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
# Open a file to save the responses
with open(os.path.join(script_dir, 'output', 'mistral.txt'), 'w', encoding='utf-8') as response_file:
    sample_number = 0
    for title, description in tqdm(zip(titles, descriptions), total=len(titles), desc="Processing samples"):
        title = title.strip()
        description = description.strip()
        # description = ''
        sample_number += 1

        valid_response = False
        retries = 0
        while not valid_response and retries < 5:
            prompt = f"""
            [INST] 
            Given an ad listing with the title '{title}' and description '{description}', answer the follwing questions without any explanation or extra text.
            Identify the species mentioned in the text, including specific names, e.g., 'Nile crocodile' instead of just 'crocodile'.
            Select the product type from the following options: Animal fibers, Animal parts (bone or bone-like), Animal parts (fleshy), Coral product, Egg, Extract, Food, Ivory products, Live, Medicine, Nests, Organs and tissues, Powder, Scales or spines, Shells, Skin or leather products, Taxidermy, Insects.
            The response should be in the format:
            "Product Type: [type]
            Species: [species]"
            [/INST]
            """
            if retries > 0:  # Adjust the prompt for retries
                prompt = f"Please provide a valid product type from the given options. {prompt}"

            # Set the desired number of new tokens to generate
            num_new_tokens = 120
            num_prompt_tokens = len(tokenizer(prompt)['input_ids'])
            max_length = num_prompt_tokens + num_new_tokens
            result = gen(prompt, max_length=max_length)  # Assuming gen is defined to generate text
            generated_text = result[0]['generated_text'].replace(prompt, '').strip()

            # Check if the generated product type is valid
            valid_response = check_product_type(generated_text)
            retries += 1

        # Format and save the generated text to the file
        response_file.write(f"Sample_{sample_number}\n{generated_text}\n\n")



print("Responses generated and saved to mistral.txt.")


# import re

# def process_and_clean_responses(input_file, output_file):
#     fields_to_capture = [
#         "Product Type:",
#         "Species:"
#     ]

#     def remove_parentheses(text):
#         # Remove text within parentheses (non-greedy match)
#         return re.sub(r'\(.*?\)', '', text).strip()

#     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#         sample_data = []
#         captured_fields = set()  # Track captured fields for the current sample

#         for line in infile:
#             if line.startswith("Sample_"):
#                 if sample_data:
#                     outfile.write('\n'.join(sample_data) + '\n\n')
#                     sample_data = []
#                     captured_fields = set()
#                 sample_data.append(line.strip())
#             else:
#                 for field in fields_to_capture:
#                     if field in line and field not in captured_fields:
#                         clean_line = re.sub(r"^\d+\.\s*", "", line).strip()
#                         # Apply parentheses removal for all fields except "Species"
#                         # if "Species:" not in line:
#                         #     clean_line = remove_parentheses(clean_line)
#                         sample_data.append(clean_line)
#                         captured_fields.add(field)
#                         break

#         if sample_data:
#             outfile.write('\n'.join(sample_data) + '\n')

# # Specify the input and output file paths
# input_file_path = 'wilddesc/responses_mistral.txt'
# output_file_path = 'wilddesc/responses_cleaned_mistral.txt'

# # Process the responses
# process_and_clean_responses(input_file_path, output_file_path)

# print("Responses have been cleaned and saved to 'responses_cleaned_mistral.txt'.")



import re

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Splitting the file content by 'Sample_' to separate samples
        samples = [sample.strip() for sample in file.read().split('Sample_') if sample.strip()]
        data = {}
        for sample in samples:
            header, *lines = sample.split('\n')  # Splitting each sample into lines
            fields = {}
            for line in lines:
                parts = line.split(': ')
                if len(parts) >= 2:  # Ensuring there's both a key and a value
                    fields[parts[0]] = parts[1]
                else:
                    # Optionally handle lines that don't match the expected format
                    print(f"Warning: Skipping line due to unexpected format: {line}")
            data[header] = fields
    return data


# Load the cleaned responses and ground truth data
responses = load_data(os.path.join(script_dir, 'output', 'mistral.txt'))
ground_truths = load_data(os.path.join(script_dir, 'data', 'ground_infer.txt'))



def validate_species(response, ground_truth):
    # Normalize and split both response and ground truth into sets of words
    response_words = {word.lower() for word in re.split(r'\W+', response)}
    ground_truth_words = {word.lower() for word in re.split(r'\W+', ground_truth)}

    # Check for any common words between the sets
    overlap = response_words & ground_truth_words

    # Return 1 if there's any overlap, otherwise 0
    return 1 if overlap else 0


def validate_product_type(response, ground_truth):
    response_types = set(response.split(', '))
    ground_truth_types = set(ground_truth.split(', '))
    return 1 if response_types & ground_truth_types else 0

def validate_field(response, ground_truth):
    return 1 if response.strip() == ground_truth.strip() else 0

def validate_responses(responses, ground_truths):
    # Define the fields you are interested in
    field_scores = {field: [] for field in ["Product Type", "Species"]}
    detailed_scores = {}

    for sample, fields in responses.items():
        ground = ground_truths.get(sample, {})
        sample_scores = []
        for field, response in fields.items():
            if field.strip() in field_scores:  # Strip spaces and check if the field is recognized
                ground_value = ground.get(field.strip(), "")  # Strip spaces for uniformity
                if field.strip() == "Species":
                    score = validate_species(response, ground_value)
                elif field.strip() == "Product Type":
                    score = validate_product_type(response, ground_value)
                else:
                    score = validate_field(response, ground_value)
                sample_scores.append(score)
                field_scores[field.strip()].append(score)  # Use the stripped field for indexing
        detailed_scores[sample] = sum(sample_scores) / len(sample_scores) if sample_scores else 0

    # Calculate average scores for each field
    average_scores_by_field = {field: sum(scores)/len(scores) if scores else 0 for field, scores in field_scores.items()}
    # Calculate the final average score across all fields
    final_average_score = sum(average_scores_by_field.values()) / len(average_scores_by_field) if average_scores_by_field else 0

    return detailed_scores, average_scores_by_field, final_average_score



# Validate the responses against the ground truths
detailed_scores, average_scores_by_field, final_average_score = validate_responses(responses, ground_truths)

# Output detailed scores and averages for debugging
print("Detailed Scores by Sample:")
for sample, score in detailed_scores.items():
    print(f"{sample}: {score:.2f}")
print("\nAverage Scores by Field:")
for field, avg_score in average_scores_by_field.items():
    print(f"{field}: {avg_score:.2f}")
print(f"\nFinal Average Score Across All Fields: {final_average_score:.2f}")

# with open('https __cites.org_eng_app_appendices.php.htm', 'r', encoding='utf-8') as file:
#     html_content = file.read()
    
# soup = BeautifulSoup(html_content, 'html.parser')

# samples = []
# with open('wilddesc/responses_cleaned_mistral.txt', 'r') as f:
#     sample = {}
#     for line in f:
#         line = line.strip()
#         if line.startswith('Sample_'):
#             if sample:  # If there's a sample collected, add it to the list
#                 samples.append(sample)
#             sample = {"Name": line}  # Start a new sample
#         elif line.startswith('Product Type:'):
#             sample['Product Type'] = line.split(':', 1)[1].strip()
#         elif line.startswith('Species:'):
#             species_info = line.split(':', 1)[1].strip()
#             wildlife_family = species_info.split('(')[-1].split(')')[0]
#             suspected_species = species_info.split(' (')[0].strip()
#             sample['Wildlife Family'] = wildlife_family
#             sample['Suspected Species'] = suspected_species
#     if sample:  # Don't forget to add the last sample
#         samples.append(sample)


# def get_cites_status(species_name):
#     return find_matches_and_species_info(soup, species_name)

# with open('wilddesc/responses_detailed_mistral.txt', 'w') as output_file:
#     for sample in samples:
#         # Check if 'Wildlife Family' is missing or empty for the sample
#         if 'Wildlife Family' not in sample or not sample['Wildlife Family'].strip():
#             continue  # Skip this sample
        
#         cites_status = get_cites_status(sample['Wildlife Family'])
#         formatted_output = f"{sample['Name']}\n"
#         formatted_output += f"Product Type: {sample['Product Type']}\n"
#         formatted_output += f"Wildlife Family: {sample['Wildlife Family']}\n"
#         formatted_output += f"Suspected Species: {sample['Suspected Species']}\n"
#         formatted_output += "Trade Status:\n"
#         formatted_output += cites_status + "\n\n"
#         output_file.write(formatted_output)

# def summarize_cites_status(detailed_filename, summary_filename):
#     with open(detailed_filename, 'r') as detailed_file:
#         detailed_content = detailed_file.read()

#     samples = detailed_content.split('Sample_')[1:]  # Splitting each sample, skipping the first empty split
#     summaries = []

#     for sample in samples:
#         lines = sample.strip().split('\n')
#         sample_id = 'Sample_' + lines[0].split('\n')[0].strip()  # Re-adding 'Sample_' prefix
#         product_type = lines[1].split(':', 1)[1].strip()
#         wildlife_family = lines[2].split(':', 1)[1].strip()
#         suspected_species = lines[3].split(':', 1)[1].strip()
        
#         # Initialize an empty list to collect CITES statuses
#         cites_statuses = []

#         # Check each Appendix and add the status to cites_statuses list if found
#         if 'Appendix I:' in sample:
#             cites_statuses.append("I")
#         if 'Appendix II:' in sample:
#             cites_statuses.append("II")
#         if 'Appendix III:' in sample:
#             cites_statuses.append("III")

#         # Determine the summarized trade status based on the collected statuses
#         if cites_statuses:
#             cites_status = "Possible CITES Appendix " + "/".join(cites_statuses)
#         else:
#             cites_status = "Not in CITES"

#         summary = f"{sample_id}\nProduct Type: {product_type}\nWildlife Family: {wildlife_family}\nSuspected Species: {suspected_species}\nTrade Status: {cites_status}\n"
#         summaries.append(summary)

#     # Writing the summary to a new file
#     with open(summary_filename, 'w') as summary_file:
#         summary_file.write("\n".join(summaries))



# summarize_cites_status('wilddesc/responses_detailed_mistral.txt', 'wilddesc/responses_short_mistral.txt')
