import json
import random
import os
from collections import defaultdict
from itertools import combinations
DATA_DIR = 'val2014'
THRESHOLD = 50



question_list = {}
non_numeric = []
folder_pairs = defaultdict(list)
only_pair = []

potential_folder_pairs = defaultdict(list)
potential_only_pair = []

error = []
for folder in os.scandir(DATA_DIR):
    if folder.is_dir():
        info = json.load(open(f'{folder.path}/info.json', 'r'))
        question = info['question']
        filtered_data = []
        for i in info['imgId2ans']:
            for sample in i:
                if not i[sample].isnumeric():
                    non_numeric.append((i, folder.name))
                if i[sample].isnumeric() and int(i[sample]) > 2:
                    file = f'{folder.path}/{sample}.jpg'
                    if os.path.isfile(file):
                        filtered_data.append((file, int(i[sample])))
                    else:
                        error.append(file)
    if len(filtered_data) > 1:
        question_list[folder.name] = question
        original_pairs = list(combinations(filtered_data, 2))

        random.shuffle(original_pairs)
        pairs = original_pairs[:THRESHOLD]
        potential_pairs = original_pairs[THRESHOLD:]
        for i in pairs:
            i = list(i)
            random.shuffle(i)
            if i[0][1] > i[1][1]:
                answer = 'Left'
            elif i[0][1] < i[1][1]:
                answer = "Right"
            else:
                answer = 'Same'
            folder_pairs[folder.name].append((i[0][0], i[1][0], answer, folder.name))
            only_pair.append((i[0][0], i[1][0], answer, folder.name))

        for i in potential_pairs:
            i = list(i)
            random.shuffle(i)
            if i[0][1] > i[1][1]:
                answer = 'Left'
            elif i[0][1] < i[1][1]:
                answer = "Right"
            else:
                answer = 'Same'
            potential_folder_pairs[folder.name].append((i[0][0], i[1][0], answer, folder.name))
            potential_only_pair.append((i[0][0], i[1][0], answer, folder.name))

with open('vqa_val2014_pairs.json', 'w') as fout:
    json.dump(only_pair, fout)

with open('vqa_val2014_pairs_folder.json', 'w') as fout:
    json.dump(folder_pairs, fout)

with open('vqa_val2014_potential_pairs.json', 'w') as fout:
    json.dump(potential_only_pair, fout)

with open('vqa_val2014_potential_pairs_folder.json', 'w') as fout:
    json.dump(potential_folder_pairs, fout)

with open('vqa_val2014_question.json', 'w') as fout:
    json.dump(question_list, fout)

print()










