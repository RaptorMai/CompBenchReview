from setup import GPT
import os
import json
from collections import defaultdict
from collections import Counter
from copy import deepcopy

model = 'gemini'
output_file = [f'results/soccernet/germany_bundesliga_{model}.json',
               f'results/soccernet/europe_uefa-champions-league_{model}.json',
               f'results/soccernet/england_epl_{model}.json',
               f'results/soccernet/france_ligue-1_{model}.json',
               f'results/soccernet/spain_laliga_{model}.json',
               f'results/soccernet/italy_serie-a_{model}.json',
               ]



summary_model = GPT()

for league_file in output_file:
    summary_file = league_file.replace(f'{model}.json', f'{model}_summary.json')
    if os.path.isfile(summary_file):
        continue

    league_result = json.load(open(league_file))
    league_new = []
    for i in league_result:
        tmp = deepcopy(i)
        prompt = f'''
        I ask an AI model a question about two frames of a football match. The question is “These are two frames in a soccer match. Which frame happens first?” The text below is the answer from the AI model. But the answer is too long. I only want either “Left” or “Right” as the final answer. Based on this answer, please return either “Left” or “Right”, which can reflect what the AI model wants to answer. Note that the model may use “frame 1" to refer to “Left” and “frame 2" to refer to “Right”; “first frame” to refer to “Left” and “second frame” to refer to “Right”. They also can use left and right directly. Your task is to summarize the answer below as “Left” or “Right”.
        Here are some examples:
        Answer: The frame on the right happens first.In the frame on the right, the player with the ball is further away from the goal. In the frame on the left, the player with the ball is closer to the goal. This means that the player in the frame on the right had to have taken the shot from further away, and therefore it happened first.
        Return: Right
        Answer: The first frame happens first. It shows Lewandowski getting ready to strike the ball with his Left foot. The second frame shows Lewandowski after he has kicked the ball with his Left foot.
        Return: Left
        Answer: The second frame happens first.In the first frame, the player in white is about to throw the ball. In the second frame, the player in white has released the ball. So the second frame happens first.
        Return: Right
        Answer: Frame 2In the first frame, we can see that the referee is holding the ball. In the second frame, the referee has thrown the ball to the player. So we can conclude that the second frame happens first.
        Return: Right
        Examples end.
        Answer: {i['gemini_answer']}
        What’s the return?
        '''
        summary_answer = summary_model.inference_text(prompt)
        print(i['gemini_answer'])
        print(f'summary: {summary_answer}')
        print('-----------------')
        tmp['gemini_summary_answer'] = summary_answer
        league_new.append(tmp)
    with open(summary_file, 'w') as fout:
        json.dump(league_new, fout)
