import json
import random
import os
from collections import defaultdict
from itertools import combinations
from setup import GPT


DATA_DIR = '/local/scratch/jihyung/comp_imgs/dataset/vqav2/counting_images/train2014/'
THRESHOLD = 50

prompt = '''
I want to turn a question about count in an image into a question comparing the counts in two images. Here are some examples:

"How many panes are on the windows?" becomes "Which image has more panes on the windows?"
"How many different numbers are shown?" becomes "Which image has more different numbers?"
"How many bananas are there?" becomes "Which image has more bananas?"
"How many people is in the canoe?" becomes "Which image has more people in the canoe?"
"How many whiskers does the dog have?" becomes "Which image has more whiskers on the dog?"
"How many pieces of artwork are on the wall behind the men?" becomes "Which image has more pieces of artwork on the wall behind the men?"
"How many drawers are in the center of the television console?" becomes "Which image has more drawers in the center of the television console?"
"How many of these people are in the process of brushing their teeth?" becomes "Which image has more people in the process of brushing their teeth?"
"How many recycle containers does the house across the street have?" becomes "Which image has more recycle containers at the house across the street?"

Based on these example, would you transform the following question? Please only return the transformed question without other words.

'''


model = GPT()

non_numeric = []
folder_pairs = defaultdict(list)
cap_pair = []
all_pair = []


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
        actual_prompt = prompt + question
        updated_question = model.inference_text(actual_prompt)
        print(f'q: {question}, new: {updated_question}')
        original_pairs = list(combinations(filtered_data, 2))

        random.shuffle(original_pairs)
        pairs = original_pairs[:THRESHOLD]

        for i in pairs:
            tmp = {}
            i = list(i)
            random.shuffle(i)
            if i[0][1] > i[1][1]:
                answer = 'Left'
            elif i[0][1] < i[1][1]:
                answer = "Right"
            else:
                answer = 'Same'
            tmp['image_1'] = i[0][0]
            tmp['image_1'] = i[1][0]
            tmp['answer'] = answer
            tmp['folder'] = folder.name
            tmp['question'] = updated_question

            cap_pair.append(tmp)


with open('vqa_train_pairs.json', 'w') as fout:
    json.dump(cap_pair, fout)

#
# with open('vqa_val2014_pairs.json', 'w') as fout:
#     json.dump(only_pair, fout)
#
# with open('vqa_val2014_pairs_folder.json', 'w') as fout:
#     json.dump(folder_pairs, fout)
#
# with open('vqa_val2014_potential_pairs.json', 'w') as fout:
#     json.dump(potential_only_pair, fout)
#
# with open('vqa_val2014_potential_pairs_folder.json', 'w') as fout:
#     json.dump(potential_folder_pairs, fout)
#
# with open('vqa_val2014_question.json', 'w') as fout:
#     json.dump(question_list, fout)
#
# print()










