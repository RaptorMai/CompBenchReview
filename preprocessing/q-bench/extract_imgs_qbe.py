import json
import argparse
import os
from collections import defaultdict, Counter
import copy
import shutil
import sys

############################## Configuration #####################################
parser=argparse.ArgumentParser(description='Extract images from Q-bench')
parser.add_argument('--root_dir',type=str,default="/local/scratch/jihyung/comp_imgs/dataset/q-bench", help='directory of Q-bench dataset')
parser.add_argument('--split',type=str,default="dev", help='name of dataset split')


import datasets

config = parser.parse_args()

def save_imgs(config):
    

    # Annotations
    anns = json.load(open(os.path.join(config.root_dir, f"llvisionqa_{config.split}.json")))

    que2ans = defaultdict(list)

    for ann in anns:
        que = ann["question"]
        que2ans[que].append(ann)


    print("que list: ", que2ans.keys())
    print("len(que): ", len(que2ans.keys()))

    print(que2ans["What is the blur level of the image?"])


    sys.exit(0)


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



def main():

    save_imgs(config)


if __name__ == '__main__':
    main()

