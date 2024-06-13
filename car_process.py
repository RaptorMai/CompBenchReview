# from setup import setup_gemini
from collections import defaultdict
import google.generativeai as genai
import time
from PIL import Image as PIL_Image
from collections import Counter
from setup import compute_clip_similarity
import json
import os
from os.path import join
import random
from os import listdir


###########filter pair
filename = 'car_label/car_clip.json'
output = 'car_label/car_sorted_5000.json'
top_k = 5000

with open(filename, 'r') as file:
    try:
        raw_data = json.load(file)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON")


sorted_data = sorted(raw_data, key = lambda x: -x[2])

process_car = []
for idx in range(top_k):
    tmp = {}
    '/local/scratch/jihyung/comp_imgs/dataset/comp_cars/data/test_image/97/872/2009/c9fb135eef7588.jpg'
    pair = sorted_data[idx]
    old_year = pair[0].split('/')[-2]
    old = pair[0].replace('/local/scratch/jihyung/comp_imgs/dataset/comp_cars/data/', 'comp_cars/')
    new_year = pair[1].split('/')[-2]
    new = pair[1].replace('/local/scratch/jihyung/comp_imgs/dataset/comp_cars/data/', 'comp_cars/')
    rand_int = random.randint(0, 1)
    if rand_int == 0:
        tmp['image_1'] = old
        tmp['image_2'] = new
        tmp['answer'] = 'Right'
    else:
        tmp['image_1'] = new
        tmp['image_2'] = old
        tmp['answer'] = 'Left'
    process_car.append(tmp)

with open(output, 'w') as fout:
    json.dump(process_car, fout)









########generate pair

# data_path = '/local/scratch/jihyung/comp_imgs/dataset/comp_cars/data/test_image/'
# filename = 'car_clip.json'
#
#
# all_brand = [join(data_path, i) for i in listdir(data_path) if i != '.DS_Store']
#
# all_pairs = []
# sub_list = []
# c = []
# tmp = {}
# clip_L, clip_L_preprocess = clip.load("ViT-L/14", device='cuda')
# for brand in all_brand:
#     for car in os.scandir(brand):
#         if car.name == '.DS_Store':
#             continue
#         years = sorted([int(i) for i in listdir(car) if i.isdigit()])
#         if len(years) == 1 or years[-1] - years[0] < 3:
#             continue
#         for image_1 in listdir(join(car.path, str(years[0]))):
#             for image_2 in listdir(join(car.path, str(years[-1]))):
#                 image_1_path = join(car.path, str(years[0]), image_1)
#                 image_2_path = join(car.path, str(years[-1]), image_2)
#                 c_score = compute_clip_similarity(image_1_path, image_2_path, clip_L, clip_L_preprocess)
#                 print(image_1_path)
#                 print(image_2_path)
#                 print(c_score)
#                 pair = (image_1_path, image_2_path, c_score)
#                 all_pairs.append(pair)
#
#
# with open(filename, 'w') as fout:
#     json.dump(all_pairs, fout)