import json
from collections import Counter
from copy import deepcopy
import random

input_file = f'Labeled_data/Wildfish_label/wildfish_0_60_500sample.json'
output_file = f'Labeled_data/Wildfish_label/wildfish_0_60_500sample_rebalanced.json'

with open(input_file, 'r') as file:
    try:
        data = json.load(file)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON")

gt = []
update_data = []
for idx, pair in enumerate(data):
    if pair:
        if pair['answer'] in ('Left', 'Right'):
            update_data.append(pair)
            gt.append(pair['answer'])

left = gt.count('Left')
right = gt.count('Right')
assert left + right == len(gt)

target_left = len(gt) // 2
target_right = len(gt) - target_left

cur_left = 0
cur_right = 0

new = []
for idx, pair in enumerate(update_data):
    if pair:
        if pair['answer'] == 'Left':
            if cur_left < target_left:
                new.append(pair)
                cur_left += 1
            else:
                tmp = deepcopy(pair)
                tmp['image_1'] = pair['image_2']
                tmp['image_2'] = pair['image_1']
                tmp['answer'] = 'Right'
                new.append(tmp)
                cur_right += 1

        elif pair['answer'] == 'Right':
            if cur_right < target_right:
                new.append(pair)
                cur_right += 1
            else:
                tmp = deepcopy(pair)
                tmp['image_1'] = pair['image_2']
                tmp['image_2'] = pair['image_1']
                tmp['answer'] = 'Left'
                new.append(tmp)
                cur_left += 1

random.shuffle(new)

print(new[:10])
with open(output_file, 'w') as fout:
    json.dump(new, fout)
