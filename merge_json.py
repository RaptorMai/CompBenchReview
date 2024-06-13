import os
import json

label_folder = 'Labeled_data/mb_label'
# prefix = 'Checked'
output_path = f'{label_folder}/MB_all.json'
file_list = ['MB_val_0_315.json', 'MB_test_0_400.json', 'MB_test_400_612.json']


final = []
for filename in os.listdir(label_folder):
    # Check if the file is a JSON file by its extension
    if filename.endswith(".json") and filename in file_list: #and filename.startswith(prefix): # in file_list: #filename.startswith(prefix):
        # Construct full file path
        file_path = os.path.join(label_folder, filename)
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
                print(f"Successfully read {filename}:")
                final += data
            except json.JSONDecodeError:
                print(f"Failed to decode JSON from {filename}")

with open(output_path, 'w') as fout:
    json.dump(final, fout)