import json
import random
import os
from setup import setup_gpt, gpt_inference_text

INFERENCE_NUM = 50

only_pair = json.load(open('vqa_val2014_pairs.json'))
folder_pairs = json.load(open('vqa_val2014_pairs_folder.json'))
question_list = json.load(open('vqa_val2014_question.json'))
(folder_id, folder_question) = zip(*question_list.items())


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


model = setup_gpt()

updated_questions = {}

for i in question_list:
    question = question_list[i]
    print(f'Processing batch {i}')
    actual_prompt = prompt + question
    answer = gpt_inference_text(actual_prompt, model)
    updated_questions[i] = answer

for i in only_pair:
    i.append(updated_questions[i[3]])

with open('vqa_val2014_pairs_updated_question.json', 'w') as fout:
    json.dump(only_pair, fout)


only_pair = json.load(open('vqa_val2014_pairs_updated_question.json'))
ret = []

for pair in only_pair:
    new = {}
    new['image_1'] = pair[0]
    new['image_2'] = pair[1]
    new['answer'] = pair[2]
    new['folder'] = pair[3]
    new['question'] = pair[4]
    ret.append(new)

with open('vqa_val2014_pairs_updated_questions.json', 'w') as fout:
    json.dump(ret, fout)