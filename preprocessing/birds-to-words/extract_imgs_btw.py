import json
import argparse
import os
from collections import defaultdict, Counter
import copy
import shutil
import sys

############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract images from brids-to-words')
parser.add_argument('--root_dir',type=str,default="/local/scratch/jihyung/comp_imgs/dataset/birds-to-words", help='directory of birds-to-words dataset')
parser.add_argument('--split',type=str,default="val", help='name of dataset split')


import datasets

config = parser.parse_args()

def save_imgs(config):
    

    # Annotations
    anns = json.load(open(os.path.join(config.root_dir, f"{config.split}_ann.json")))


    # save images by object_attribute
    cnt=0
    for _, ann in enumerate(anns):

        smp_id = ann["id"]

        img1_id = ann["images"][0]["path"]
        img2_id = ann["images"][1]["path"]
        que = ann["conversation"][0]["content"]
        ans = ann["conversation"][1]["content"]

        info = {
            "img1": img1_id.split(".jpg")[0],
            "img2": img2_id.split(".jpg")[0],
            "que": que,
            "ans": ans
        }

        out_f = os.path.join(config.root_dir, "pair_images", config.split, str(smp_id))
        if not os.path.exists(out_f):
            os.makedirs(out_f)

        # get images
        img1_pth = os.path.join(config.root_dir, f"{config.split}_images", img1_id)
        img2_pth = os.path.join(config.root_dir, f"{config.split}_images", img2_id)

        # store two images
        shutil.copy2(img1_pth, os.path.join(out_f, img1_id)) 
        shutil.copy2(img2_pth, os.path.join(out_f, img2_id)) 
        
        with open(os.path.join(out_f, "info.json"), "w") as fp:
            json.dump(info, fp)

        cnt+=1


    print(f"# pairs: {cnt}")

def load_save_ann(config):

    # https://huggingface.co/datasets/TIGER-Lab/Mantis-Instruct/tree/main/birds-to-words
    # Requirments: python 3.10, datasets==2.19.1

    dataset = datasets.load_dataset("TIGER-Lab/Mantis-Instruct", "birds-to-words")
    dataset = list(dataset[config.split])
    with open(os.path.join(config.root_dir, f"{config.split}_ann.json"), "w") as fp:
        json.dump(dataset, fp)
    

def main():

    # load and save annotation files
    # load_save_ann(config)

    # Save two images into a folder
    save_imgs(config)


if __name__ == '__main__':
    main()

