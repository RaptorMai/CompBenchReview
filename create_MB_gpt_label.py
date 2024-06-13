import json
from setup import compute_clip_similarity, encode_image, gpt_inference_one, setup_gpt
import clip

###########
data_path = 'magic_brush/dev'
clip_threhold = 0.94
#######


f = open(f'{data_path}/edit_turns.json')
instructions = json.load(f)
f = open(f'{data_path}/global_descriptions.json')
global_descriptions = json.load(f)
f = open(f'{data_path}/local_descriptions.json')
local_descriptions = json.load(f)

len(instructions)
clip_L, clip_L_preprocess = clip.load("ViT-L/14", device='cuda')
count = 0
data = []
for edit_idx, sample in enumerate(instructions):
    data.append({"path_input": f'{data_path}/images/{sample["input"].split("-")[0]}/{sample["input"]}',
                 "input_global": f'{global_descriptions[sample["input"].split("-")[0]][sample["input"]]}',
                 "path_output": f'{data_path}/images/{sample["input"].split("-")[0]}/{sample["output"]}',
                 "output_local": f'{local_descriptions[sample["input"].split("-")[0]][sample["output"]]}',
                 "output_global": f'{global_descriptions[sample["input"].split("-")[0]][sample["output"]]}',
                 "instruction": sample['instruction'],
                 "CLIP_similarity": compute_clip_similarity(
                     f'{data_path}/images/{sample["input"].split("-")[0]}/{sample["input"]}',
                     f'{data_path}/images/{sample["input"].split("-")[0]}/{sample["output"]}', model=clip_L,
                     preprocess=clip_L_preprocess)}
                )
    count += 1

sorted_data = sorted(data, key=lambda d: d['CLIP_similarity'], reverse=True)

headers = setup_gpt()
prompt = '''
I want to extract as many components as possible from the provided images. Component examples are shown below. However, components are not limited to the following components. Please only provide the component name without any explanation and separate the component names with commons. If a human or an animal is shown in the images and hair, eye, hand, mouth, ear, and leg, etc. are visible, ensure to include them, Similarly, try to find all the components as detailed as possible. 
1. leg, 2. eye, 3. ear, 4. food, 5. pillow, 6. flower, 7. plate, 8. window, 9. door, 10. chair, 11. dining table, 12. sofa, 13. banana, 14. bowl, 15. sugar, 16. blender, 17. berry, 18. lizard, 19. watermelon, 20. motorcycle, 21. apple, 22. curtain, 23, cookies, 24, cake, 25. hair, 26, hat, 27, dresses, 28. bacon, 29. butter, 30, jam, 31, bread 32, surfboard, 33, t-shirt, 34, pants, 35, hands, 36. fridge, 37, plants, 38. cabinet, 39, sink, 40, car, 41, girl, 42, boy
'''

option_data = []
for idx, pair in enumerate(sorted_data):
    if pair['CLIP_similarity'] >= 0.94:
        answer = gpt_inference_one(pair['path_output'], prompt, headers)
        if not isinstance(answer, str):
            print('error')
        pair['GPT_option'] = answer
    else:
        pair['GPT_option'] = None
    option_data.append(pair)
    print(idx)

with open('magic_brush_val_94clip.json', 'w') as fout:
    json.dump(option_data, fout)