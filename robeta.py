import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaConfig
from transformers import DataCollatorWithPadding
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from captum.attr import IntegratedGradients, visualization
import os

nltk.download('stopwords')
nltk.download('wordnet')

script_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Function to clean and preprocess text
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Read the files
title_finetune = read_file(os.path.join(script_dir, 'data', 'title_finetune.txt'))
description_finetune = read_file(os.path.join(script_dir, 'data', 'description_finetune.txt'))
ground_finetune = read_file(os.path.join(script_dir, 'data', 'ground_finetune.txt'))

title_infer = read_file(os.path.join(script_dir, 'data', 'title_infer.txt'))
description_infer = read_file(os.path.join(script_dir, 'data', 'description_infer.txt'))
ground_infer = read_file(os.path.join(script_dir, 'data', 'ground_infer.txt'))

# Process the ground truth files
product_types_finetune = process_ground_truth(ground_finetune)
product_types_infer = process_ground_truth(ground_infer)

# Combine title and description into a single input for the classifier
inputs_finetune = [f"{title.strip()} {description.strip()}" for title, description in zip(title_finetune, description_finetune)]
inputs_infer = [f"{title.strip()} {description.strip()}" for title, description in zip(title_infer, description_infer)]

# Preprocess the combined inputs
inputs_finetune = [preprocess_text(text) for text in inputs_finetune]
inputs_infer = [preprocess_text(text) for text in inputs_infer]

# Create DataFrames for fine-tuning and inference
finetune_data = pd.DataFrame({
    'input': inputs_finetune,
    'product_type': product_types_finetune
})

infer_data = pd.DataFrame({
    'input': inputs_infer,
    'product_type': product_types_infer
})

# Combine product types from both datasets for fitting the label encoder
all_product_types = list(finetune_data['product_type']) + list(infer_data['product_type'])

# Initialize the tokenizer and model
model_dir = '/tmp/roberta/' 
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir=model_dir)
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=18, cache_dir=model_dir)  # 18 product types


model.to(device)

# Encode the labels
label_encoder = LabelEncoder()
label_encoder.fit(all_product_types)
finetune_data['product_type'] = label_encoder.transform(finetune_data['product_type'])
infer_data['product_type'] = label_encoder.transform(infer_data['product_type'])

# Tokenize the datasets
train_encodings = tokenizer(list(finetune_data['input']), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(infer_data['input']), truncation=True, padding=True, max_length=512)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the dataset objects
train_dataset = Dataset(train_encodings, list(finetune_data['product_type']))
test_dataset = Dataset(test_encodings, list(infer_data['product_type']))

# Training arguments
training_args = TrainingArguments(
    output_dir='/tmp/results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='/tmp/logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    max_steps=1000,
    save_strategy="no",
)

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc
    }
    


# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer)
)

# Train the model
trainer.train()

# Evaluate the model
# eval_results = trainer.evaluate()
# print(eval_results)




# Predictions for the inference set
predictions = trainer.predict(test_dataset)
predicted_labels = label_encoder.inverse_transform(predictions.predictions.argmax(axis=1))

# Writing output to file
with open(os.path.join(script_dir, 'output', 'roberta.txt'), 'w') as file:
    for idx, (input_text, predicted_label) in enumerate(zip(inputs_infer, predicted_labels), 1):
        file.write(f"Sample_{idx}\n")
        file.write(f"Product Type: {predicted_label}\n")
        file.write(f"Species:\n\n")  # leaving species blank as requested

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
responses = load_data(os.path.join(script_dir, 'output', 'roberta.txt'))
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
        # print(ground_truths)
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