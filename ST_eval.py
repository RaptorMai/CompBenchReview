from eval_manager import EvalManager
import os
import json

model = 'gemini'
source_data = f'ST_label/State_Transform_merged0_0_1000.json'
output_file = f'results/ST/0_1000_{model}.json'


prompt = ''' If you choose the first image, return First and if you choose the second image, return Second.
Please only return either First or Second without any other words, spaces or punctuation. '''

class EvalManagerST(EvalManager):
    def eval(self, prompt, output_name):
        results = []
        # {"image_1": "val2014/1982/408449.jpg", "image_2": "val2014/1982/218189.jpg", "answer": "Right", "folder": "1982", "question": "Which image has more people walking?"}
        for pair in self.data:
            if pair:
                actual_prompt = pair['question'] + prompt
                answer = self.inf_model.inference_two(pair['image_1'], pair['image_2'], actual_prompt)
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


eval = EvalManagerST(model)
eval.setup()
eval.load_data(source_data)


eval.eval(prompt, output_file)
