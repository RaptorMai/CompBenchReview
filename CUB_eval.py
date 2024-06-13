from eval_manager import EvalManager
import os
import json
import os.path
import base64
from copy import deepcopy

model = 'gemini'
source_data = f'CUB_label/CUB_0_1000_rebalanced.json'
output_file = f'results/CUB/0_1000_{model}.json'


prompt = ''' If you choose the first image, return First and if you choose the second image, return Second.
Please only return either Second or First without any other words, spaces or punctuation. '''

# prompt = ''' If you choose the left image, return Left and if you choose the right image, return Right.
# Please only return either Left or Right without any other words, spaces or punctuation. '''

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# fix cub path
# with open('CUB_label/CUB_0_1000_rebalanced.json', 'r') as json_file:
#     cub_raw = json.load(json_file)
#
# new = []
# for idx, pair in enumerate(cub_raw):
#     if pair:
#         tmp = deepcopy(pair)
#         image_1 = pair['image_1'].replace("\\", '/')
#         image_2 = pair['image_2'].replace("\\", '/')
#         if image_1.startswith('att_cls_images'):
#             image_1 = 'cub_200_2011/' + image_1
#         if image_2.startswith('att_cls_images'):
#             image_2 = 'cub_200_2011/' + image_2
#         try:
#             en_1 = encode_image(image_1)
#             en_2 = encode_image(image_2)
#         except:
#             print(image_1, image_2)
#         tmp['image_1'] = image_1
#         tmp['image_2'] = image_2
#         new.append(tmp)
#
#
# with open('CUB_label/CUB_0_1000_rebalanced_fixed_path.json', 'w') as fout:
#     json.dump(new, fout)



class EvalManagerCUB(EvalManager):
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
                        if pair['image_1']==i['image_1'] and pair['image_2']==i['image_2']:
                            if f'{self.model}_answer' in i:
                                results.append(i)
                                find = True
                                break
                if not find:
                    actual_prompt = pair['question'] + prompt
                    #print(pair['image_1'], pair['image_2'])
                    image_1 = pair['image_1'].replace("\\", '/')
                    image_2 = pair['image_2'].replace("\\", '/')
                    if image_1.startswith('att_cls_images'):
                        image_1 = 'cub_200_2011/' + image_1
                    if image_2.startswith('att_cls_images'):
                        image_2 = 'cub_200_2011/' + image_2
                    # test before run
                    # try:
                    #     en_1 = encode_image(image_1)
                    #     en_2 = encode_image(image_2)
                    # except:
                    #     print(image_1, image_2)

                    answer = self.inf_model.inference_two(image_1, image_2, actual_prompt)
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
        #

eval = EvalManagerCUB(model)
eval.setup()
eval.load_data(source_data)


eval.eval(prompt, output_file)
