import os
import json
import random
from copy import deepcopy


label_files = [
    'Labeled_data/car_label/car_sorted_5000.json',
    'Labeled_data/celebA_label/celebA_0_141_rebalanced.json',
    'Labeled_data/cub_label/CUB_0_1000_rebalanced_fixed_path.json',
    'Labeled_data/depth_label/depth_0_529_rebalanced.json',
    'Labeled_data/fashion_label/fashion_0_179.json',
    'Labeled_data/fer2013_label/fer2013_0_720_rebalanced.json',
    'Labeled_data/soccernet_label/soccer_all.json',
    'Labeled_data/vqav2_label/vqa_val2014_pairs_updated_questions_cap_50.json',
    'Labeled_data/Wildfish_label/wildfish_0_60_500sample_rebalanced.json',
    'Labeled_data/qbench2_label/qbench2_source.json',
    'Labeled_data/mb_label/MB_all.json',
    'Labeled_data/spot_difference_label/surveillance_0_1270.json',
    'Labeled_data/vaw_label/VAW_0_3133.json',
    'Labeled_data/st_label/State_Transform_merged0_0_1000.json',
    # 'Labeled_data/st_label/State_Transform_merged0_0_1000_att.json',
    # 'Labeled_data/st_label/State_Transform_merged0_0_1000_state.json',
# 'Labeled_data/vaw_label/VAW_0_3133_att.json',
# 'Labeled_data/vaw_label/VAW_0_3133_state.json',
]

total_val = []
total_test = []
total = []
for label in label_files:
    name = label.split('/')[1]
    label_file = label.split('/')[-1]

    label_test = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_test.json'
    label_val = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_val.json'

    test_data = json.load(open(label_test))
    val_data = json.load(open(label_val))

    total_val += val_data
    total_test += test_data

    # total += [i for i in json.load(open(label)) if i]
    print(label)
    print(len(val_data) + len(test_data))
    # print(len([i for i in json.load(open(label)) if i]))

print(f'test: {len(total_test)}')
print(f'val: {len(total_val)}')
print(f'{len(total_test) + len(total_val)}')
# print(f'total: {len(total)}')