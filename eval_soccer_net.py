import argparse
import json
from setup import compute_clip_similarity, GEMINI, GPT
import clip
import os
import random
import time

FOLDER = ('1_frames_actions', '2_frames_actions')
ACTION = ["Indirect free-kick", "Throw-in", "Foul", "Shots off target", "Shots on target", "Goal", "Corner", "Direct free-kick", "Penalty"]
INTERVAL = 2
CLIP_THRESHOLD = 0.78

def inference_main():
    args = setup_parser().parse_args()
    results = {}
    if args.server_run:
        data_folder = '/local/scratch/jihyung/comp_imgs/dataset/soccernet/' + args.data_folder
    else:
        data_folder = args.data_folder
    if args.model == 'gpt':
        inf_model = GPT()
    elif args.model == 'gemini':
        inf_model = GEMINI()
    # else:
    #     raise NotImplementedError

    clip_L, clip_L_preprocess = clip.load("ViT-L/14", device='cuda')
    for match in os.scandir(data_folder):
        match_results = []
        if match.is_dir():
            for folder in os.scandir(match):
                if folder.name in FOLDER:
                    pair_list = list(os.scandir(folder))
                    for pair in pair_list:
                        if pair.name.split("_")[1] in ACTION:
                            if args.model == 'gemini':
                                time.sleep(5)
                            tmp = {}
                            first = f'{pair.path}/{pair.name.split("_")[0]}.png'
                            second_index = "{:0{width}d}".format(int(pair.name.split("_")[0]) + INTERVAL,
                                                                 width=len(pair.name.split("_")[0]))
                            second = f'{pair.path}/{second_index}.png'
                            c_score = compute_clip_similarity(first, second, clip_L, clip_L_preprocess)
                            if c_score < CLIP_THRESHOLD:
                                continue

                            rand_1 = random.randint(0, 1)

                            if rand_1 == 0:
                                tmp['left'] = first
                                tmp['right'] = second
                                tmp['answer'] = 'Left'
                            else:
                                tmp['left'] = second
                                tmp['right'] = first
                                tmp['answer'] = 'Right'
                            tmp['CLIP'] = c_score
                            tmp['time'] = pair.name.split("_")[0]
                            tmp['action'] = pair.name.split("_")[1]
                            # prompt = f'''
                            #            These are two frames related to {pair.name.split("_")[1]} in a soccer match.
                            #            Which frame happens first? Please only return one option from (First, Second, None)
                            #            without any other words. If these two frames are exactly the same, select None.
                            #            Otherwise, choose First if the first frame happens first and select Second
                            #            if the second frame happens first.
                            #            '''
                            # prompt = f'''
                            # These are two frames related to {pair.name.split("_")[1]} in a soccer match.
                            # Which frame happens first? Please only return one option from (Left, Right, None)
                            # without any other words. If these two frames are exactly the same, select None.
                            # Otherwise, choose Left if the first frame happens first and select Right
                            # if the second frame happens first.
                            # '''
                            prompt = f'''
                            These are two frames related to {pair.name.split("_")[1]} in a soccer match. Which frame happens first?  
                            Return either Frame 2 or Frame 1 without explanation and any other words.
                            '''
                            # prompt = f''' which frame happens first? Choose from Left or Right or Same. Only return one of these three words without other words.'''
                            #answer = inf_model.inference_two(tmp['left'], tmp['right'], prompt)
                            answer = 'test'
                            if args.model == 'gemini':
                                with open('gemini_tracker.json', 'w') as fout:
                                    json.dump(inf_model.all_keys, fout)
                                if answer == 'quotas':
                                    print('exceed quotas, try it next day')
                            tmp[f'{args.model}_answer'] = answer
                            match_results.append(tmp)
                            print(tmp)
        results[match.name] = match_results
    filename = f'results/soccernet/{args.data_folder}/results_{args.model}.json'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fout:
        json.dump(results, fout)

def setup_parser():
    parser = argparse.ArgumentParser(description='SoccerNet.')
    parser.add_argument('--data_folder', type=str, default='england_epl/2014-2015')
    parser.add_argument('--server_run', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='gpt')
    return parser

if __name__ == '__main__':
    inference_main()