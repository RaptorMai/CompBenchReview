from eval_manager import EvalManager
import os
import json

model = 'gpt'
source_data = f'Labeled_data/mb_label/MB_all.json'
output_file = f'results/MB/results_MB_val_test_decompose.json'



class EvalManagerMB(EvalManager):
    def eval(self, prompt, output_name, decompose):
        results = []
        prev_result = None
        # {"image_1": "val2014/1982/408449.jpg", "image_2": "val2014/1982/218189.jpg", "answer": "Right", "folder": "1982", "question": "Which image has more people walking?"}
        if decompose:
            if os.path.isfile(output_name):
                prev_result = json.load(open(output_name, 'r'))

            for idx, pair in enumerate(self.data):
                if pair:
                    find = False
                    if prev_result:
                        for i in prev_result:
                            if pair['path_input'] == i['path_input'] and pair['path_output'] == i['path_output']:
                                if f'{self.model}_decompose_answer' in i:
                                    results.append(i)
                                    find = True
                                    break
                    if not find:
                        decompose_question = f'''
                        Please describe the image as detailed as possible based on the following options: {pair['Modified_option']}.
                        The description should include the objects in the options and their properties or features in adjectives, such as color, size, state, etc.
                        '''

                        answer_1 = self.inf_model.inference_one(pair['path_input'], decompose_question)
                        answer_2 = self.inf_model.inference_one(pair['path_output'], decompose_question)

                        final_question = f'These are descriptions for two images. Left images: {answer_1}. ' \
                                         f'Right image: {answer_2}. Based on the descriptions, What is the most obvious difference between two images?' \
                                         f'Choose from the following options. If there is no obvious difference, choose None. ' \
                                         f'Options: None,{pair["Modified_option"]}.' \
                                         f'Please only return one of the options without any other words.'
                        answer = self.inf_model.inference_text(final_question)
                        print(f'{self.model} decompose answer: {answer}, {pair["Answeer"]}')
                        pair[f'{self.model}_decompose_answer'] = answer
                        results.append(pair)
        else:
            for pair in self.data:
                actual_prompt  = f'''
                What is the most obvious difference between two images? Choose from the following options. 
                If there is no obvious difference, choose None. Options: None,{pair['Modified_option']}.
                Please only return one of the options without any other words. 
                '''
                answer = self.inf_model.inference_two(pair['path_input'], pair['path_output'], actual_prompt)
                pair[f'{self.model}_answer'] = answer
                results.append(pair)
                print(f'{self.model}_answer: {answer}, gt: {pair["Answeer"]}')
                if self.model == 'gemini':
                    with open('gemini_tracker.json', 'w') as fout:
                        json.dump(self.inf_model.all_keys, fout)
                    if answer == 'quotas':
                        print('exceed quotas, try it next day')
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        with open(output_name, 'w') as fout:
            json.dump(results, fout)


eval = EvalManagerMB(model)
eval.setup()
eval.load_data(source_data)


eval.eval('', output_file, True)
