import os
import json
import random
from copy import deepcopy
from string import punctuation

random.seed(30)
split_ratio = 0.8


def get_acc(key, content):
    error = []
    refine = []
    for pair in content:
        new = deepcopy(pair)
        if 'Right' in pair[key] or 'Second' in pair[key]:
            new[key] = 'Right'
            refine.append(new)
        elif 'Left' in pair[key] or 'First' in pair[key]:
            new[key] = 'Left'
            refine.append(new)
        elif 'Same' in pair[key]:
            new[key] = 'Same'
            refine.append(new)
        else:
            error.append(pair)
            print(pair)
    correct_pred = []
    wrong_pred = []
    correct = []
    wrong = []

    for pair in refine:
        if pair['answer'] == pair[key]:
            correct.append(pair)
            correct_pred.append(pair[key])
        elif pair['answer'] != pair[key]:
            wrong.append(pair)
            wrong_pred.append(pair[key])
        else:
            print('fuck')
    print('Acc')
    print(len(correct) / (len(correct) + len(wrong) + len(error)))
    print('Error')
    print(len(error) / (len(correct) + len(wrong) + len(error)))



def get_acc_option(key, gt_key, data):
    correct = []
    wrong = []
    invalid = []
    error = []
    c = 0
    for i in data:
        c += 1
        if isinstance(i[key], dict):
            print(i)
            error.append(i)
        else:
            pred = i[key].lstrip(' ').rstrip(' ').lower().strip(punctuation)
        gt = i[gt_key].lstrip(' ').rstrip(' ').lower().strip(punctuation)
        #options = [j.lstrip(' ').rstrip(' ').lower().strip(punctuation) for j in i['option'].split(',')] + ['none']
        options = [j.lstrip(' ').rstrip(' ').lower().strip(punctuation) for j in i['option']] + ['none']
        if 'error' in pred:
            print('error')
            print(i)
            error.append(i)
        elif pred not in options:
            print('invalid')
            print(i)
            invalid.append(i)
        else:
            if pred == gt:
                correct.append(i)
            else:
                wrong.append(i)
    print('Acc')
    print(len(correct) / (len(correct) + len(wrong) + len(error) + len(invalid)))
    print('Error')
    print(len(error) / (len(correct) + len(wrong) + len(error) + len(invalid) ))
    print('Invalid')
    print(len(invalid) / (len(correct) + len(wrong) + len(error) + len(invalid)))
    return correct, wrong, invalid, error, c




result_match = {
    'car_label': 'car',
    'celebA_label': 'celebA',
    'cub_label': 'cub',
    'depth_label': 'depth',
    'fashion_label': 'fashion',
    'fer2013_label': 'fer2013',
    'mb_label': 'mb',
    'qbench2_label': 'qbench2',
    'soccernet_label': 'soccernet',
    'spot_difference_label': 'spot_diff',
    'st_label': 'st',
    'vaw_label': 'vaw',
    'vqav2_label': 'vqav2',
    'Wildfish_label': 'wildfish',
}



option_label_file = [
    # 'Labeled_data/qbench2_label/qbench2_source.json',
    'Labeled_data/mb_label/MB_all.json',
    #'Labeled_data/spot_difference_label/surveillance_0_1270.json',
]

for label in option_label_file:
    name = label.split('/')[1]
    label_file = label.split('/')[-1]
    label_test = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_test.json'
    label_val = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_val.json'
    result_folder = result_match[name]
    print(name, result_folder)
    data = [i for i in json.load(open(label, 'r')) if i]
    split_num = int(len(data) * split_ratio)
    random.shuffle(data)
    test = data[:split_num]
    val = data[split_num:]
    for file in os.scandir(f'results/{result_folder}/'):
        if 'decompose' in file.name or 'DS' in file.name or os.path.isdir(file):
            continue
        model = file.name.split('.')[0].split('_')[-1]
        print(model)
        if file.name.endswith('json'):
            results = json.load(open(file, 'r'))
        else:
            results = [json.loads(q) for q in open(os.path.expanduser(file), "r")]
        test_result = []
        val_result = []
        fast = {}
        key = f'{model}_answer'
        if 'mb' in file.name:
            image1 = 'path_input'
            image2 = 'path_output'
            gt_key = 'Answeer'
        elif 'sur' in label:
            image1 = 'image_1'
            image2 = 'image_2'
            gt_key = 'Answer'
        elif 'bench' in label:
            image1 = 'image'
            image2 = 'image'
            gt_key = 'answer'
        else:
            image1 = 'image_1'
            image2 = 'image_2'
            gt_key = 'answer'
        for idx, i in enumerate([k for k in results if isinstance(k, dict)]):

            if 'question' in i:
                if (i[image1], i[image2], i['question'], i[gt_key]) in fast:
                    print(i)
                fast[(i[image1], i[image2], i['question'], i[gt_key])] = i
            else:
                if(i[image1], i[image2], i[gt_key]) in fast:
                    print(i)
                fast[(i[image1], i[image2], i[gt_key])] = i
        for i in test:
            try:
                if 'question' in i:
                    test_result.append(fast[(i[image1], i[image2], i['question'],  i[gt_key])])
                else:
                    test_result.append(fast[(i[image1], i[image2], i[gt_key])])
            except:
                i[gt_key] = 'Error'
                test_result.append(i)

        for i in val:
            try:
                if 'question' in i:
                    val_result.append(fast[(i[image1], i[image2], i['question'],  i[gt_key])])
                else:
                    val_result.append(fast[(i[image1], i[image2], i[gt_key])])
            except:
                i[gt_key] = 'Error'
                val_result.append(i)
        print('val')
        get_acc_option(key, gt_key, val_result)
        print('test')
        get_acc_option(key, gt_key, test_result)
        print('---------------------------------------------')

    with open(label_test, 'w') as fout:
        json.dump(test, fout)
    with open(label_val, 'w') as fout:
        json.dump(val, fout)

# state_att_file = [
#     'Labeled_data/st_label/State_Transform_merged0_0_1000.json',
#     'Labeled_data/vaw_label/VAW_0_3133.json',
# ]
#
# attribute = ['Size', 'Color', 'Pattern', 'Texture', 'Shape']
#
# for label in state_att_file:
#     name = label.split('/')[1]
#     label_file = label.split('/')[-1]
#
#     label_test_state = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_state_test.json'
#     label_val_state = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_state_val.json'
#
#     label_test_att = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_att_test.json'
#     label_val_att = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_att_val.json'
#
#     result_folder = result_match[name]
#     print(name, result_folder)
#
#     data_att = [i for i in json.load(open(label, 'r')) if i and i['type'] in attribute]
#     data_state = [i for i in json.load(open(label, 'r')) if i and i['type'] == 'State']
#
#     split_num = int(len(data_att) * split_ratio)
#     random.shuffle(data_att)
#     test_att = data_att[:split_num]
#     val_att = data_att[split_num:]
#
#     split_num = int(len(data_state) * split_ratio)
#     random.shuffle(data_state)
#     test_state = data_state[:split_num]
#     val_state = data_state[split_num:]
#
#     for file in os.scandir(f'results/{result_folder}/'):
#         if 'decompose' in file.name or 'DS' in file.name or os.path.isdir(file):
#             continue
#         model = file.name.split('.')[0].split('_')[-1]
#         print(model)
#         if file.name.endswith('json'):
#             results = json.load(open(file, 'r'))
#         else:
#             results = [json.loads(q) for q in open(os.path.expanduser(file), "r")]
#         test_state_result = []
#         val_state_result = []
#
#         test_att_result = []
#         val_att_result = []
#         key = f'{model}_answer'
#         fast = {}
#         image1 = 'image_1'
#         image2 = 'image_2'
#         for idx, i in enumerate([k for k in results if isinstance(k, dict)]):
#             if 'question' in i:
#                 if (i[image1], i[image2], i['question'], i['answer']) in fast:
#                     print(i)
#                 fast[(i[image1], i[image2], i['question'], i['answer'])] = i
#             else:
#                 if(i[image1], i[image2], i['answer']) in fast:
#                     print(i)
#                 fast[(i[image1], i[image2], i['answer'])] = i
#
#         for i in test_att:
#             try:
#                 if 'question' in i:
#                     test_att_result.append(fast[(i[image1], i[image2], i['question'],  i['answer'])])
#                 else:
#                     test_att_result.append(fast[(i[image1], i[image2], i['answer'])])
#             except:
#                 i[key] = 'Error'
#                 test_att_result.append(i)
#
#         for i in test_state:
#             try:
#                 if 'question' in i:
#                     test_state_result.append(fast[(i[image1], i[image2], i['question'],  i['answer'])])
#                 else:
#                     test_state_result.append(fast[(i[image1], i[image2], i['answer'])])
#             except:
#                 i[key] = 'Error'
#                 test_state_result.append(i)
#
#
#         for i in val_att:
#             try:
#                 if 'question' in i:
#                     val_att_result.append(fast[(i[image1], i[image2], i['question'],  i['answer'])])
#                 else:
#                     val_att_result.append(fast[(i[image1], i[image2], i['answer'])])
#             except:
#                 i[key] = 'Error'
#                 val_att_result.append(i)
#
#         for i in val_state:
#             try:
#                 if 'question' in i:
#                     val_state_result.append(fast[(i[image1], i[image2], i['question'],  i['answer'])])
#                 else:
#                     val_state_result.append(fast[(i[image1], i[image2], i['answer'])])
#             except:
#                 i[key] = 'Error'
#                 val_state_result.append(i)
#
#         print('att')
#         print('val')
#         get_acc(key, val_att_result)
#         print('test')
#         get_acc(key, test_att_result)
#         print('state')
#         print('val')
#         get_acc(key, val_state_result)
#         print('test')
#         get_acc(key, test_state_result)
#         print('---------------------------------------------')
#
#     with open(label_test_state, 'w') as fout:
#         json.dump(test_state, fout)
#     with open(label_val_state, 'w') as fout:
#         json.dump(val_state, fout)
#
#     with open(label_test_att, 'w') as fout:
#         json.dump(test_att, fout)
#     with open(label_val_att, 'w') as fout:
#         json.dump(val_att, fout)


# label_files = [
#     'Labeled_data/car_label/car_sorted_5000.json',
#     'Labeled_data/celebA_label/celebA_0_141_rebalanced.json',
#     'Labeled_data/cub_label/CUB_0_1000_rebalanced_fixed_path.json',
#     'Labeled_data/depth_label/depth_0_529_rebalanced.json',
#     'Labeled_data/fashion_label/fashion_0_179.json',
#     'Labeled_data/fer2013_label/fer2013_0_720_rebalanced.json',
#     'Labeled_data/soccernet_label/soccer_all.json',
#     'Labeled_data/vqav2_label/vqa_val2014_pairs_updated_questions_cap_50.json',
#     'Labeled_data/Wildfish_label/wildfish_0_60_500sample_rebalanced.json',
# ]
#
# for label in label_files:
#     name = label.split('/')[1]
#     label_file = label.split('/')[-1]
#     label_test = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_test.json'
#     label_val = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_val.json'
#     result_folder = result_match[name]
#     print(name, result_folder)
#     data = [i for i in json.load(open(label, 'r')) if i]
#     split_num = int(len(data) * split_ratio)
#     random.shuffle(data)
#     test = data[:split_num]
#     val = data[split_num:]
#     for file in os.scandir(f'results/{result_folder}/'):
#         if 'decompose' in file.name or 'DS' in file.name or os.path.isdir(file):
#             continue
#         model = file.name.split('.')[0].split('_')[-1]
#         print(model)
#         if file.name.endswith('json'):
#             results = json.load(open(file, 'r'))
#         else:
#             results = [json.loads(q) for q in open(os.path.expanduser(file), "r")]
#         test_result = []
#         val_result = []
#         fast = {}
#         key = f'{model}_answer'
#         if 'soccer' in file.name:
#             image1 = 'left'
#             image2 = 'right'
#             if model == 'gemini':
#                 key = 'gemini_summary_answer'
#         else:
#             image1 = 'image_1'
#             image2 = 'image_2'
#         for idx, i in enumerate([k for k in results if isinstance(k, dict)]):
#             if 'cub' in file.name:
#                 image_1 = i[image1].replace("\\", '/')
#                 image_2 = i[image2].replace("\\", '/')
#                 if image_1.startswith('att_cls_images'):
#                     image_1 = 'cub_200_2011/' + image_1
#                 if image_2.startswith('att_cls_images'):
#                     image_2 = 'cub_200_2011/' + image_2
#                 i[image1] = image_1
#                 i[image2] = image_2
#
#             if 'question' in i:
#                 if (i[image1], i[image2], i['question'], i['answer']) in fast:
#                     print(i)
#                 fast[(i[image1], i[image2], i['question'], i['answer'])] = i
#             else:
#                 if(i[image1], i[image2], i['answer']) in fast:
#                     print(i)
#                 fast[(i[image1], i[image2], i['answer'])] = i
#         for i in test:
#             try:
#                 if 'question' in i:
#                     test_result.append(fast[(i[image1], i[image2], i['question'],  i['answer'])])
#                 else:
#                     test_result.append(fast[(i[image1], i[image2], i['answer'])])
#             except:
#                 i[key] = 'Error'
#                 test_result.append(i)
#
#         for i in val:
#             try:
#                 if 'question' in i:
#                     val_result.append(fast[(i[image1], i[image2], i['question'],  i['answer'])])
#                 else:
#                     val_result.append(fast[(i[image1], i[image2], i['answer'])])
#             except:
#                 i[key] = 'Error'
#                 val_result.append(i)
#         print('val')
#         get_acc(key, val_result)
#         print('test')
#         get_acc(key, test_result)
#         print('---------------------------------------------')
#
#     with open(label_test, 'w') as fout:
#         json.dump(test, fout)
#     with open(label_val, 'w') as fout:
#         json.dump(val, fout)










