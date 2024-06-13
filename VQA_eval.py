from eval_manager import EvalManager
import os
import json

model = 'gemini'

class EvalManagerVQA(EvalManager):
    def eval(self, prompt, output_name):
        results = []
        # {"image_1": "val2014/1982/408449.jpg", "image_2": "val2014/1982/218189.jpg", "answer": "Right", "folder": "1982", "question": "Which image has more people walking?"}
        for pair in self.data:
            actual_prompt = pair['question'] + prompt
            answer = self.inf_model.inference_two(pair['image_1'], pair['image_2'], actual_prompt)
            pair[f'{self.model}_answer'] = answer
            results.append(pair)
            print(f'{self.model}_answer: {answer}, {pair["answer"]}')
            if self.model ==  'gemini':
                with open('gemini_tracker.json', 'w') as fout:
                    json.dump(self.inf_model.all_keys, fout)
                if answer == 'quotas':
                    print('exceed quotas, try it next day')
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        with open(output_name, 'w') as fout:
            json.dump(results, fout)


eval = EvalManagerVQA(model)
eval.setup()
eval.load_data(f'vqa_val2014_pairs_updated_questions_cap_50.json')

prompt = '''
If the second image has more, return Right. If the first image has more, return Left. . 
If both images have the same number, return Same. Please only return either Left or Right or Same without any other words, spaces or punctuation. 
'''

eval.eval(prompt, f'results/VQA_2014/results_cap_50_{model}.json')
