import pandas as pd
import re
from tqdm import tqdm
import ahocorasick
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Function to read the input files and return lists of lines
def read_file(filename):
    with open(filename, 'r') as file:
        return file.readlines()

# Function to process the ground truth file and extract the product type
def process_ground_truth(ground_lines):
    product_types = []
    for line in ground_lines:
        if line.startswith("Product Type:"):
            product_type = line.split(":")[1].strip().split(',')[0]
            product_types.append(product_type)
    return product_types

def match_product_type(text, product_dict):
    for product, keywords in product_dict.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                return product
    return "Unknown"


# Function to build the Aho-Corasick automaton for species matching
def load_species_into_automaton(species_file):
    """Load species names into an Aho-Corasick automaton."""
    automaton = ahocorasick.Automaton()
   
    for line in species_file:
        species_name = line.strip().lower()
        automaton.add_word(species_name, species_name)
    automaton.make_automaton()
    return automaton

def find_species_in_text(text, automaton):
    """Identify species within the given text using an Aho-Corasick automaton."""
    found_species = set()
    for end_index, original_value in automaton.iter(text.lower()):
        found_species.add(original_value)
    if found_species:
        # print(found_species)
        # print(type(found_species))
        # print(max(found_species, key=len))
        found_species = max(found_species, key=len)
        
        return found_species
    return "Unknown"


# Read the files
title_infer = read_file(os.path.join(script_dir, 'data', 'title_infer.txt'))
ground_infer = read_file(os.path.join(script_dir, 'data', 'ground_infer.txt'))

# Process the ground truth files
product_types_infer = process_ground_truth(ground_infer)

# Combine title and description into a single input
inputs_infer = [f"{title.strip()}" for title in title_infer]


# Load the species list
with open(os.path.join(script_dir, 'data', 'vernacular_names.txt'), 'r') as file:
    species_list = [line.strip() for line in file.readlines()]
    
# Build the Aho-Corasick automaton for species matching
automaton = load_species_into_automaton(species_list)

# Define the product type dictionary
product_dict = {
    'Animal fibers': ['fiber', 'fibers', 'feather', 'feathers', 'fur', 'furs'],
    'Animal parts (bone or bone-like)': ['bone', 'bones', 'teeth', 'tooth'],
    'Animal parts (fleshy)': ['fleshy'],
    'Coral product': ['coral', 'corals'],
    'Egg': ['egg', 'eggs'],
    'Extract': ['extract', 'extracts'],
    'Food': ['food', 'foods'],
    'Ivory products': ['ivory'],
    'Live': ['live'],
    'Medicine': ['medicine', 'medicines'],
    'Nests': ['nests', 'nest'],
    'Organs and tissues': ['organ','tissue', 'organs', 'tissues'],
    'Powder': ['powder', 'powders'],
    'Scales or spines': ['scale', 'spine', 'scales', 'spines'],
    'Shells': ['shell', 'shells'],
    'Skin or leather products': ['skin', 'leather', 'skins', 'leathers'],
    'Taxidermy': ['taxidermy', 'taxidermys'],
    'Insects': ['insects', 'insect']
}

# Perform string matching for product types and species
predicted_product_types = []
predicted_species = []

for text in tqdm(inputs_infer, desc="Processing"):
    predicted_product_types.append(match_product_type(text, product_dict))
    predicted_species.append(find_species_in_text(text, automaton))
    
os.makedirs(os.path.join(script_dir, 'output'), exist_ok=True)
# Writing output to file
with open(os.path.join(script_dir, 'output', 'string_match.txt'), 'w') as file:
    for idx, (input_text, predicted_label, species) in enumerate(zip(inputs_infer, predicted_product_types, predicted_species), 1):
        
        file.write(f"Sample_{idx}\n")
        file.write(f"Product Type: {predicted_label}\n")
        file.write(f"Species: {''.join(species) if species else 'Unknown'}\n\n")

print("Inference results saved.")

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
responses = load_data(os.path.join(script_dir, 'output', 'string_match.txt'))
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