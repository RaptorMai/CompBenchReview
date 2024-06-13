import os
import json
import random

label_files = [
    'Labeled_data/car_label/car_sorted_5000.json',
    'Labeled_data/celebA_label/celebA_0_141_rebalanced.json',
    'Labeled_data/cub_label/CUB_0_1000_rebalanced_fixed_path.json',
    'Labeled_data/depth_label/depth_0_529_rebalanced.json',
    'Labeled_data/fashion_label/fashion_0_179.json',
    'Labeled_data/fer2013_label/fer2013_0_720_rebalanced.json',
    'Labeled_data/mb_label/MB_val_0_315.json',
    'Labeled_data/qbench2_label/qbench2_source.json',
    'Labeled_data/soccernet_label/soccer_all.json',
    'Labeled_data/spot_difference_label/surveillance_0_1270.json',
    'Labeled_data/st_label/State_Transform_merged0_0_1000.json',
    'Labeled_data/vaw_label/VAW_0_3133.json',
    'Labeled_data/vqav2_label/vqa_val2014_pairs_updated_questions_cap_50.json',
    'Labeled_data/Wildfish_label/wildfish_0_60_500sample_rebalanced.json',
]

car_question = '''Based on these images, which car is newer in terms of its model year or release year? 
Note that this question refers solely to the year each car was first introduced or manufactured, 
not its current condition or usage.'''

DATA_DIR = '/local/scratch/jihyung/comp_imgs/dataset/'

random.seed(30)
human_eval = []
num_per_data = -1
output_path = f'human_eval_{num_per_data}.json'

for label in label_files:
    with open(label, 'r') as file:
        try:
            data = [i for i in json.load(file) if i]
            print(f"Successfully read {label}:")
            random.shuffle(data)
            candidates = data[:num_per_data]

            if 'car_sorted_5000' in label:
                for i in candidates:
                    i['question'] = car_question
                    i['final_option'] = ['Left', 'Right']
                    i['image_1'] = i['image_1'].replace('comp_cars/test_image/', 'comp_cars/data/test_image/')
                    i['image_2'] = i['image_2'].replace('comp_cars/test_image/', 'comp_cars/data/test_image/')
            elif 'MB_val_0_315' in label:
                for i in candidates:
                    i['question'] = f'''
                        What is the most obvious difference between two images? Choose from the following options. 
                        If there is no obvious difference, choose None. Options: None,{i['Modified_option']}.
                        Please only return one of the options without any other words. 
                        '''
                    i['image_1'] = i['path_input'].replace('magic_brush', 'magicbrush')
                    i['image_2'] = i['path_output'].replace('magic_brush', 'magicbrush')
                    i['answer'] = i['Answeer']
                    i['final_option'] = i['Modified_option'].split(',') + ['None']
            elif 'soccer_all' in label:
                for i in candidates:
                    i['question'] = f'''
                                These are two frames related to {i['action']} in a soccer match.
                                Which frame happens first? Please only return one option from (Left, Right, None)
                                without any other words. If these two frames are exactly the same, select None.
                                Otherwise, choose Left if the first frame happens first and select Right
                                if the second frame happens first.
                                '''
                    i['final_option'] = ['Left', 'Right', 'None']
                    i['image_1'] = 'soccernet/' + i['left']
                    i['image_2'] = 'soccernet/' + i['right']
                    i['final_option'] = ['Left', 'Right']
            elif 'qbench2_source' in label:
                for i in candidates:
                    i['image_1'] = i['image']
                    i['image_2'] = i['image']
                    i['final_option'] = i['option']
            elif 'surveillance_0_1270' in label:
                for i in candidates:
                    i['answer'] = i['Answer']
                    i['final_option'] = i['Modified_option'].split(',') + ['None']
                    i['question']  = f'''
                    What is the most obvious difference between two images? Choose from the following options. 
                    If there is no obvious difference, choose None. Options: None, {i['Modified_option']}.
                    Please only return one of the options without any other words. 
                    '''
            elif 'depth_0_529_rebalanced' in label:
                for i in candidates:
                    i['image_1'] = 'nyu_depth_v2/' + i['image_1']
                    i['image_2'] = 'nyu_depth_v2/' + i['image_2']
                    i['final_option'] = ['Left', 'Right']
            elif 'fashion_0_179' in label:
                for i in candidates:
                    i['image_1'] = 'fashionpedia/' + i['image_1']
                    i['image_2'] = 'fashionpedia/' + i['image_2']
                    i['final_option'] = ['Left', 'Right']
            elif 'State_Transform_merged0_0_1000' in label:
                for i in candidates:
                    i['image_1'] = 'transformed_states/' + i['image_1']
                    i['image_2'] = 'transformed_states/' + i['image_2']
                    i['final_option'] = ['Left', 'Right']
            elif 'VAW_0_3133' in label:
                for i in candidates:
                    i['image_1'] = 'vaw/' + i['image_1']
                    i['image_2'] = 'vaw/' + i['image_2']
                    i['final_option'] = ['Left', 'Right']
            elif 'vqa_val2014_pairs_updated_questions_cap_50' in label:
                for i in candidates:
                    i['image_1'] = 'vqav2/counting_images/' + i['image_1']
                    i['image_2'] = 'vqav2/counting_images/' + i['image_2']
                    i['final_option'] = ['Left', 'Right', 'Same']
            else:
                for i in candidates:
                    i['final_option'] = ['Left', 'Right']

            # print(candidates[0]['question'])
            print(candidates[0]['final_option'])
            # print(candidates[0]['answer'])
            print(candidates[0]['image_1'], candidates[0]['image_2'])
            # print('ok')
            print('------------------------------------')
            human_eval += candidates
        except Exception as e:
            print(f"Failed {label}: {e}")

for i in human_eval:
    i['image_1'] = DATA_DIR + i['image_1']
    i['image_2'] = DATA_DIR + i['image_2']

with open(output_path, 'w') as fout:
    json.dump(human_eval, fout)

# for i in human_eval:
#     if os.path.isfile(i['image_1']) and os.path.isfile(i['image_1']):
#         continue
#     else:
#         print(i['image_1'], i['image_2'])