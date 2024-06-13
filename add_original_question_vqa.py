import os
import json

label_folder = 'Labeled_data/vqav2_label'
# prefix = 'Checked'
file_path = f'{label_folder}/vqa_val2014_pairs_updated_questions_cap_50.json'

with open(file_path, 'r') as file:
    try:
        data = json.load(file)
        print(f"Successfully read {file_path}:")
        for i in data:

    except json.JSONDecodeError:
        print(f"Failed to decode JSON from {file_path}")




# with open(output_path, 'w') as fout:
#     json.dump(final, fout)