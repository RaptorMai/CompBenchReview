import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, merge_images
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

import sys


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def prompt_for_choose(options, args):
    que = args.q_prompt.replace("[options_var]", options)

    return que

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, args):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.args = args

    def __getitem__(self, index):
        line = self.questions[index]

        #image_file = line["image"]
        img1 = line["path_input"]
        img2 = line["path_output"]
        two_img_links = [os.path.join(self.image_folder, img1), os.path.join(self.image_folder, img2)]

        #print(f"index: {index}, two_img_links: {two_img_links}")

        options = line["Modified_option"]
        qs = prompt_for_choose(options, self.args)

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        #print("prompt: ", prompt)

        if not (os.path.isfile(two_img_links[0]) and os.path.isfile(two_img_links[1])):
            # one of image paths is wrong, so return dummy values

            image_tensor = torch.zeros((4,3,336,336))
            input_ids = torch.zeros(49)
            img_size = (1290,480)

            return input_ids, image_tensor, img_size   

        else:
            # merge two images
            image = merge_images(two_img_links)
            
            #image.save(f"merged_img_smps/mb_merged_{index}.jpg")

            # single image
            #image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

            image_tensor = process_images([image], self.image_processor, self.model_config)[0]

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

            

            return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, args, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, args)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)

    #questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = json.load(open(args.question_file))

    # skip empy sample
    ques_tmp = []
    for smp in questions:
        if len(smp) == 0:
            continue
        else:
            ques_tmp.append(smp)
    questions = ques_tmp
    print(f"# total samples: {len(questions)}")

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    pred_file = os.path.expanduser(args.pred_file)
    os.makedirs(os.path.dirname(pred_file), exist_ok=True)
    ans_file = open(pred_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args)

    wrong_pths = []
    idx = 0
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        cur_path_input = line["path_input"]
        cur_input_global = line["input_global"]
        cur_path_output = line["path_output"]
        cur_output_local = line["output_local"]
        cur_output_global = line["output_global"]
        cur_CLIP_similarity = line["CLIP_similarity"]
        cur_GPT_option = line["GPT_option"]
        cur_instruction = line["instruction"]
        cur_If_modified_option = line["If_modified_option"]
        cur_Modified_option = line["Modified_option"]
        cur_Answeer = line["Answeer"]
        cur_Quality = line["Quality"]


        #print("input_ids!!: ", input_ids.shape)
        # one of paths is wrong. Skip inference on this sample
        if torch.equal(input_ids, torch.zeros((1,49))):
            ans_file.write(json.dumps({
                "path_input": cur_path_input,
                "input_global": cur_input_global,
                "path_output": cur_path_output,
                "output_local": cur_output_local,
                "output_global": cur_output_global,
                "CLIP_similarity": cur_CLIP_similarity,
                "GPT_option": cur_GPT_option,
                "instruction": cur_instruction,
                "If_modified_option": cur_If_modified_option,
                "Modified_option": cur_Modified_option,
                "Answeer": cur_Answeer,
                "Quality": cur_Quality,
                "vila_answer": "wrong path"
                }) + "\n")
            
            wrong_pths.append(idx)
            
            idx+=1
            continue

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                    "path_input": cur_path_input,
                                    "input_global": cur_input_global,
                                    "path_output": cur_path_output,
                                    "output_local": cur_output_local,
                                    "output_global": cur_output_global,
                                    "CLIP_similarity": cur_CLIP_similarity,
                                    "GPT_option": cur_GPT_option,
                                    "instruction": cur_instruction,
                                    "If_modified_option": cur_If_modified_option,
                                    "Modified_option": cur_Modified_option,
                                    "Answeer": cur_Answeer,
                                    "Quality": cur_Quality,
                                    "vila_answer": outputs
                                   }) + "\n")
        # ans_file.flush()
        idx+=1
    ans_file.close()
    print("wrong_pths: ", wrong_pths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-40b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--pred-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--q_prompt", type=str, default="")
    args = parser.parse_args()

    eval_model(args)
