from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
from datasets import Dataset
import re
import os
import argparse

parser = argparse.ArgumentParser(description="T5 Model Training Script")
parser.add_argument("--model", type=str, default="xl", choices=["xl", "large"],
                    help="Model Size.")

args = parser.parse_args()

seed_run = 6


torch.manual_seed(seed_run)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_run)
    torch.cuda.manual_seed_all(seed_run)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_dir = '/tmp/t5/' 
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.model == 'xl':
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl", cache_dir=model_dir)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", cache_dir=model_dir)
if args.model == 'large':
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", cache_dir=model_dir)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", cache_dir=model_dir)


script_dir = os.path.dirname(os.path.abspath(__file__))

titles_file = os.path.join(script_dir, 'data', 'title_infer.txt')
descriptions_file = os.path.join(script_dir, 'data', 'description_infer.txt')

# Read the files and pair titles with descriptions
pairs = []
with open(titles_file, 'r', encoding='utf-8') as tf, open(descriptions_file, 'r', encoding='utf-8') as df:
    titles = tf.readlines()
    descriptions = df.readlines()
    pairs = zip(titles, descriptions)  # Creates pairs of (title, description)

def create_prompts(title, description):
    # Cleaning title and description from potential newline characters
    title = title.strip()
    description = description.strip()
    

    # Creating the prompts
    species_prompt = f"Based on the following title and description, Identify the species mentioned in the text, including specific names, e.g., 'Nile crocodile' instead of just 'crocodile'.: Description: '{description}, 'Title: '{title}'."
    product_type_prompt = f"""
        Based on the following title and description, classify the product type: Title: '{title}'. Description: '{description}'.
        Select the product type from the following options: Animal fibers, Animal parts (bone or bone-like), Animal parts (fleshy), Coral product, Egg, Extract, Food, Ivory products, Live, Medicine, Nests, Organs and tissues, Powder, Scales or spines, Shells, Skin or leather products, Taxidermy, Insects.
        The product type is:
        """
        
  
   
    return species_prompt, product_type_prompt



os.makedirs(os.path.join(script_dir, 'output'), exist_ok=True)
with open(os.path.join(script_dir, 'output', 't5.txt'), 'w', encoding='utf-8') as output_file:
    for i, (title, description) in enumerate(pairs, start=1):
        species_prompt, product_type_prompt = create_prompts(title, description)
        
        # Process species identification
        input_ids = tokenizer(species_prompt, return_tensors="pt").input_ids.to(device)
        species_outputs = model.generate(input_ids, max_new_tokens = 15)
        species_prediction = tokenizer.decode(species_outputs[0], skip_special_tokens=True)
        
        # Process product type classification
        input_ids = tokenizer(product_type_prompt, return_tensors="pt").input_ids.to(device)
        product_outputs = model.generate(input_ids, max_new_tokens = 15)
        product_prediction = tokenizer.decode(product_outputs[0], skip_special_tokens=True)
        
        
        # Write the formatted output to file
        output_file.write(f"Sample_{i}\n")
        output_file.write(f"Product Type: {product_prediction}\n")
        output_file.write(f"Species: {species_prediction}\n\n")
        
print("Responses generated and saved to t5.txt.")


import re

def process_and_clean_responses(input_file, output_file):
    fields_to_capture = [
        "Product Type:",
        "Species:"
    ]

    def remove_parentheses(text):
        # Remove text within parentheses (non-greedy match)
        return re.sub(r'\(.*?\)', '', text).strip()

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        sample_data = []
        captured_fields = set()  # Track captured fields for the current sample

        for line in infile:
            if line.startswith("Sample_"):
                if sample_data:
                    outfile.write('\n'.join(sample_data) + '\n\n')
                    sample_data = []
                    captured_fields = set()
                sample_data.append(line.strip())
            else:
                for field in fields_to_capture:
                    if field in line and field not in captured_fields:
                        clean_line = re.sub(r"^\d+\.\s*", "", line).strip()
                        # Apply parentheses removal for all fields except "Species"
                        # if "Species:" not in line:
                        #     clean_line = remove_parentheses(clean_line)
                        sample_data.append(clean_line)
                        captured_fields.add(field)
                        break

        if sample_data:
            outfile.write('\n'.join(sample_data) + '\n')

# Specify the input and output file paths

input_file_path = os.path.join(script_dir, 'output', 't5.txt')
output_file_path = os.path.join(script_dir, 'output', 't5_cleaned.txt')

# Process the responses
process_and_clean_responses(input_file_path, output_file_path)

print("Responses have been cleaned and saved to 't5_cleaned.txt'.")



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
responses = load_data(os.path.join(script_dir, 'output', 't5.txt'))
ground_truths = load_data(os.path.join(script_dir, 'data', 'ground_infer.txt'))



def validate_species(response, ground_truth):
    # Normalize and split both response and ground truth into sets of words
    response_words = {word.lower() for word in re.split(r'\W+', response) if word.strip()}
    ground_truth_words = {word.lower() for word in re.split(r'\W+', ground_truth) if word.strip()}

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
    field_scores = {field: [] for field in ["Product Type", "Species"]}
    detailed_scores = {}

    for sample, fields in responses.items():
        ground = ground_truths.get(sample, {})
        sample_scores = []
        for field, response in fields.items():
            ground_value = ground.get(field, "")
            if field == "Species":
                score = validate_species(response, ground_value)
            elif field == "Product Type":
                score = validate_product_type(response, ground_value)
            else:
                score = validate_field(response, ground_value)
            sample_scores.append(score)
            field_scores[field].append(score)  # Aggregate scores by field
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