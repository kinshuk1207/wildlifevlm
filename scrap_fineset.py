#create the finetune splits of the text files 

import random
random.seed(66)

def combine_datasets(title_files, description_files, ground_files, output_title, output_description, output_ground):
    combined_titles = []
    combined_descriptions = []
    combined_grounds = []
    
    # Process each set of files
    sample_index = 1
    for title_file, description_file, ground_file in zip(title_files, description_files, ground_files):
        with open(title_file, 'r') as f:
            titles = f.read().strip().split('\n')
        with open(description_file, 'r') as f:
            descriptions = f.read().strip().split('\n')
        with open(ground_file, 'r') as f:
            grounds = f.read().strip().split('Sample_')[1:]  # Ignore the first empty entry if any

        # Append each title and description, ensuring not to skip any entries
        combined_titles.extend(titles)
        combined_descriptions.extend(descriptions)

        # Process and renumber each ground entry
        for ground in grounds:
            ground_content = 'Sample_{}\n'.format(sample_index) + ground.split('\n', 1)[1].strip()
            if not ground_content.endswith('\n'):
                ground_content += '\n'
            combined_grounds.append(ground_content)
            sample_index += 1

    # Save combined files
    with open(output_title, 'w') as f:
        f.write('\n'.join(combined_titles) + '\n')
    with open(output_description, 'w') as f:
        f.write('\n'.join(combined_descriptions) + '\n')
    with open(output_ground, 'w') as f:
        f.write('\n'.join(combined_grounds) + '\n')

    return 'Datasets combined and saved successfully!'



def create_finetuning_datasets(title_file, description_file, ground_file, percentage):
    # Read the files
    with open(title_file, 'r') as f:
        titles = f.readlines()
    with open(description_file, 'r') as f:
        descriptions = f.readlines()
    with open(ground_file, 'r') as f:
        grounds = ''.join(f.readlines()).split('Sample_')[1:]  # Splitting directly on 'Sample_' and ignoring the first empty entry
    
    # Prepare data and calculate class distribution
    samples = []
    product_type_count = {}
    for index, ground in enumerate(grounds):
        # Getting product types from each sample
        product_types = ground.split('\n')[1].replace('Product Type: ', '').split(', ')
        sample_index = int(ground.split('\n')[0])
        for product_type in product_types:
            if product_type.strip() not in product_type_count:
                product_type_count[product_type.strip()] = []
            product_type_count[product_type.strip()].append(sample_index)
    
    # Determine number of samples per product type for fine-tuning
    fine_tuning_samples = []
    for product_type, indices in product_type_count.items():
        k = int(len(indices) * (percentage / 100))
        fine_tuning_samples.extend(random.sample(indices, k))
    
    # Split into fine-tuning and inference sets
    title_finetune, title_infer = [], []
    desc_finetune, desc_infer = [], []
    ground_finetune, ground_infer = [], []
    
    fine_index, infer_index = 1, 1
    for i in range(len(titles)):
        ground_content = grounds[i].split('\n', 1)[1]  # Skip the initial number
        if i in fine_tuning_samples:
            title_finetune.append(titles[i])
            desc_finetune.append(descriptions[i])
            ground_finetune.append(f'Sample_{fine_index}\n{ground_content}')
            fine_index += 1
        else:
            title_infer.append(titles[i])
            desc_infer.append(descriptions[i])
            ground_infer.append(f'Sample_{infer_index}\n{ground_content}')
            infer_index += 1
    
    
    # Save new files
    with open('/N/u/kisharma/Quartz/wilddesc/full_new/title_finetune.txt', 'w') as f:
        f.writelines(title_finetune)
    with open('/N/u/kisharma/Quartz/wilddesc/full_new/title_infer.txt', 'w') as f:
        f.writelines(title_infer)
    with open('/N/u/kisharma/Quartz/wilddesc/full_new/description_finetune.txt', 'w') as f:
        f.writelines(desc_finetune)
    with open('/N/u/kisharma/Quartz/wilddesc/full_new/description_infer.txt', 'w') as f:
        f.writelines(desc_infer)
    with open('/N/u/kisharma/Quartz/wilddesc/full_new/ground_finetune.txt', 'w') as f:
        f.writelines(ground_finetune)
    with open('/N/u/kisharma/Quartz/wilddesc/full_new/ground_infer.txt', 'w') as f:
        f.writelines(ground_infer)

    return 'Datasets created successfully!'


title_files = ['wilddesc/full/title.txt', 'wilddesc/full_new/title.txt', 'wilddesc/full_new/title3.txt']
description_files = ['wilddesc/full/description.txt', 'wilddesc/full_new/description.txt', 'wilddesc/full_new/description3.txt']
ground_files = ['wilddesc/full/ground.txt', 'wilddesc/full_new/ground.txt', 'wilddesc/full_new/ground3.txt']
output_title = 'wilddesc/full_new/combined_title.txt'
output_description = 'wilddesc/full_new/combined_description.txt'
output_ground = 'wilddesc/full_new/combined_ground.txt'

combine_datasets(title_files, description_files, ground_files, output_title, output_description, output_ground)


create_finetuning_datasets(title_file='wilddesc/full_new/combined_title.txt', description_file='wilddesc/full_new/combined_description.txt', ground_file='wilddesc/full_new/combined_ground.txt', percentage=2.5)