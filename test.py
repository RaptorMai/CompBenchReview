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
from pattern import comparative

def read_json_files(root_dir):
    results = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "results_test.json":
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for match in data:
                        results += data[match]
    return results

# Replace 'your_directory_path' with the actual path to your folder
directory_path = 'results/soccernet/germany_bundesliga'
all_data = read_json_files(directory_path)
print(all_data)