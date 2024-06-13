from setup import GPT, compute_clip_similarity
from os.path import join
import json
from os import listdir
import os


data_path = 'spot-the-diff/pair_images/test'


gpt = GPT()
prompt = '''please list all the objects and attributes associated with them, for example, black cars, people, trees, 
 white trucks, and yellow poles.  Only provide one attribute (adjective) per object. 
 Please only provide the answer without any explanation and separate the answer names with commas. '''

# all_diff = []
# for folder in list(os.scandir(data_path)):
#     all_diff += json.load(open(join(folder.path, f'sents.json')))
#
# all_text = '\n'.join(all_diff)
extract_promtp = '''these sentences describing the differences between two images. 
extract the objects from these sentences. for example, ['there is more people', 'the car moved'], you should return 
people, car.  Please only provide the answer without any explanation and separate the answer names with commas. '''



all_data = []
for idx, folder in enumerate(list(os.scandir(data_path))):
    if folder.name == '.DS_Store':
        continue
    tmp = {}
    image1 = join(folder.path, f'{folder.name}.png')
    image2 = join(folder.path, f'{folder.name}_2.png')
    tmp['image_1'] = image1
    tmp['image_2'] = image2
    tmp['image_diff'] = join(folder.path, f'{folder.name}_diff.jpg')
    tmp['difference'] = json.load(open(join(folder.path, f'sents.json')))
    diff_object = gpt.inference_text('\n'.join(tmp['difference']) + extract_promtp)
    new_prompt = prompt + 'Ensure to include these objects: ' + diff_object
    object_list_1 = gpt.inference_one(image1, new_prompt)
    object_list_2 = gpt.inference_one(image2, new_prompt)
    tmp['image_1_object_list'] = object_list_1
    tmp['image_2_object_list'] = object_list_2
    tmp['gpt_extract_objects_gt'] = diff_object
    print((idx, diff_object, object_list_1, object_list_2))
    all_data.append(tmp)



with open('surveillance_source.json', 'w') as fout:
    json.dump(all_data, fout)