import os
import json
import random
from copy import deepcopy
from string import punctuation

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
        options = [j.lstrip(' ').rstrip(' ').lower().strip(punctuation) for j in i['Modified_option'].split(',')] + ['none']
        #options = [j.lstrip(' ').rstrip(' ').lower().strip(punctuation) for j in i['option']] + ['none']
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




label_file =  'Labeled_data/vqav2_label/vqa_val2014_pairs_updated_questions_cap_50.json'

name = label_file.split('/')[1]
label_file = label_file.split('/')[-1]
label_test = 'Labeled_data/' + name + '/' + label_file.split('.')[0] + '_test.json'

test_data = json.load(open(label_test))

results = []
for file in os.scandir(f'results/vqav2/'):
    if 'finetuned' in file.name:
        results += [json.loads(q) for q in open(os.path.expanduser(file), "r")]


# result_path = 'results/depth/0_529_gpt_rebalanced_decompose.json'
#
# results = json.load(open(result_path, 'r'))

image1 = 'image_1'
image2 = 'image_2'
# image1 = 'left'
# image2 = 'right'
gt_key = 'answer'

fast = {}
test_result = []
for idx, i in enumerate([k for k in results if isinstance(k, dict)]):
    if (i[image1], i[image2], i[gt_key]) in fast:
        print(i)
    fast[(i[image1], i[image2], i[gt_key])] = i

for i in test_data:
    try:
        test_result.append(fast[(i[image1], i[image2], i[gt_key])])
    except:
        i[gt_key] = 'Error'
        test_result.append(i)

print('test')
get_acc('llava_answer', test_result)