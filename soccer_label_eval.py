from eval_manager import EvalManager
import os
import json
import os.path


model = 'gpt'
source_data = [f'Labeled_data/soccernet_label/germany_bundesliga.json',
               f'Labeled_data/soccernet_label/europe_uefa-champions-league.json',
                f'Labeled_data/soccernet_label/england_epl.json',
                f'Labeled_data/soccernet_label/france_ligue-1.json',
                f'Labeled_data/soccernet_label/spain_laliga.json',
                f'Labeled_data/soccernet_label/italy_serie-a.json',
               ]
output_file = [ f'results/soccernet/germany_bundesliga_{model}.json',
                f'results/soccernet/europe_uefa-champions-league_{model}.json',
                f'results/soccernet/england_epl_{model}.json',
                f'results/soccernet/france_ligue-1_{model}.json',
                f'results/soccernet/spain_laliga_{model}.json',
                f'results/soccernet/italy_serie-a_{model}.json',
                ]


prefix = '/local/scratch/jihyung/comp_imgs/dataset/soccernet/'

class EvalManagerSoccer(EvalManager):
    def eval(self, prompt, output_name):
        results = []
        prev_result = None
        # {"image_1": "val2014/1982/408449.jpg", "image_2": "val2014/1982/218189.jpg", "answer": "Right", "folder": "1982", "question": "Which image has more people walking?"}
        if os.path.isfile(output_name):
            prev_result = json.load(open(output_name, 'r'))
        for idx, pair in enumerate(self.data):
            if pair:
                find = False
                if prev_result:
                    for i in prev_result:
                        if pair['left']==i['left'] and pair['right']==i['right']:
                            if f'{self.model}_answer' in i:
                                results.append(i)
                                find = True
                                break
                if not find:
                    actual_prompt = f'''
                    These are two frames related to {pair['action']} in a soccer match.
                    Which frame happens first? Please only return one option from (Left, Right, None)
                    without any other words. If these two frames are exactly the same, select None.
                    Otherwise, choose Left if the first frame happens first and select Right
                    if the second frame happens first.
                    '''

                    answer = self.inf_model.inference_two(prefix + pair['left'], prefix + pair['right'], actual_prompt)
                    pair[f'{self.model}_answer'] = answer
                    results.append(pair)
                    print(f'{self.model}_answer: {answer}, {pair["answer"]}')
                    if self.model == 'gemini':
                        with open('gemini_tracker.json', 'w') as fout:
                            json.dump(self.inf_model.all_keys, fout)
                        if answer == 'quotas':
                            print('exceed quotas, try it next day')
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        with open(output_name, 'w') as fout:
            json.dump(results, fout)

for idx in range(len(source_data)):

    eval = EvalManagerSoccer(model)
    eval.setup()
    eval.load_data(source_data[idx])
    eval.eval('', output_file[idx])
