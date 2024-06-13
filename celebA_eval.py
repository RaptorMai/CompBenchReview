from eval_manager import EvalManager
import os
import json
import os.path


model = 'gpt'
source_data = f'Labeled_data/celebA_label/celebA_0_141_rebalanced.json'
output_file = f'results/celebA/0_141_{model}_decompose.json'


# prompt = ''' If you choose the first image, return First and if you choose the second image, return Second.
# Please only return either Second or First without any other words, spaces or punctuation. '''

prompt = ''' If you choose the left image, return Left and if you choose the right image, return Right.
Please only return either Left or Right without any other words, spaces or punctuation. '''

class EvalManagerCelebA(EvalManager):
    def eval(self, prompt, output_name, decompose=False):
        results = []
        prev_result = None
        # {"image_1": "val2014/1982/408449.jpg", "image_2": "val2014/1982/218189.jpg", "answer": "Right", "folder": "1982", "question": "Which image has more people walking?"}
        if decompose:
            for idx, pair in enumerate(self.data):
                decompose_question = f'Which person smiles more? ' \
                                     f'Please rate the smile from 0 to 10. 0 means no smile and 10 means smile the most ' \
                                     f'Please only return a number without any other words and punctuation.'
                answer_1 = self.inf_model.inference_one(pair['image_1'], decompose_question)
                answer_2 = self.inf_model.inference_one(pair['image_2'], decompose_question)
                try:
                    answer_1 = float(answer_1)
                    answer_2 = float(answer_2)
                    if answer_1 > answer_2:
                        answer = 'Left'
                    else:
                        answer = 'Right'
                    print(f'{self.model} decompose answer: {answer}, {pair["answer"]}')
                    pair[f'{self.model}_decompose_answer'] = answer
                except Exception as e:
                    print(e)
                    pair[f'{self.model}_decompose_answer'] = 'Error'

                results.append(pair)
        else:
            if os.path.isfile(output_name):
                prev_result = json.load(open(output_name, 'r'))
            for idx, pair in enumerate(self.data):
                if pair:
                    find = False
                    if prev_result:
                        for i in prev_result:
                            if pair['image_1']==i['image_1'] and pair['image_2']==i['image_2']:
                                if f'{self.model}_answer' in i:
                                    results.append(i)
                                    find = True
                                    break
                    if not find:
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


eval = EvalManagerCelebA(model)
eval.setup()
eval.load_data(source_data)


eval.eval(prompt, output_file, True)
