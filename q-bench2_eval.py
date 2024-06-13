from eval_manager import EvalManager
import os
import json
import os.path


model = 'gemini'
source_data = 'qbench2_label/qbench2_source.json'
output_file = f'results/qbench2/{model}.json'

#### generate source
# output_source = 'qbench2_label/qbench2_source.json'
#
# instruction = '. Please only return one of the options without any other words or punctuation.'
#
# with open('q-bench2/q-bench2-a1-dev.jsonl', 'r') as json_file:
#     json_list = list(json_file)
#
# data = []
# for json_str in json_list:
#     tmp = {}
#     result = json.loads(json_str)
#     image = 'q-bench2/' + result['img_path'].replace('\\', '/')
#     question = result['question'] + ' Options: ' + (', ').join(result['candidates']) + instruction
#     assert isinstance(result, dict)
#     tmp['image'] = image
#     tmp['question'] = question
#     tmp['answer'] = result['correct_ans']
#     tmp['option'] = result['candidates']
#     data.append(tmp)
# os.makedirs(os.path.dirname(output_source), exist_ok=True)
# with open(output_source, 'w') as fout:
#     json.dump(data, fout)


class EvalManagerQBench(EvalManager):
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
                        if pair['image']==i['image']:
                            if f'{self.model}_answer' in i:
                                results.append(i)
                                find = True
                                break
                if not find:
                    actual_prompt = pair['question']
                    answer = self.inf_model.inference_one(pair['image'], actual_prompt)
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


eval = EvalManagerQBench(model)
eval.setup()
eval.load_data(source_data)


eval.eval('', output_file)
