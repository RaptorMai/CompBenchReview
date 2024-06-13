import base64
from PIL import Image
#import torch
import requests
import google.generativeai as genai
from PIL import Image as PIL_Image
from collections import defaultdict
import time
import json
G_RPM = 15
G_RPD = 1500
# G_RPM = 3
# G_RPD = 50



class GEMINI():
    def __init__(self):
        self.all_keys = json.load(open('gemini_tracker.json'))
        self.model = genai.GenerativeModel(model_name="models/gemini-1.0-pro-vision-latest")
        #self.model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        self.current_key = None
        self.load_key()

    def load_key(self):
        for i in self.all_keys:
            if (time.time() - self.all_keys[i]['last_inf_time']) <= 60:
                if self.all_keys[i]['count_minute'] < G_RPM:
                    genai.configure(api_key=i)
                    self.current_key = i
                    return True
            elif (time.time() - self.all_keys[i]['last_inf_time']) >= 86400:
                self.all_keys[i]['count_minute'] = 0
                self.all_keys[i]['count_day'] = 0
                genai.configure(api_key=i)
                self.current_key = i
                return True
            elif (time.time() - self.all_keys[i]['last_inf_time']) > 60 and (
                    time.time() - self.all_keys[i]['last_inf_time']) <= 86400:
                if self.all_keys[i]['count_day'] < G_RPD:
                    self.all_keys[i]['count_minute'] = 0
                    genai.configure(api_key=i)
                    self.current_key = i
                    return True
        self.current_key = None
        return False

    def inference_two(self, image_1, image_2, prompt):
        try:
            img_1 = PIL_Image.open(image_1)
            img_2 = PIL_Image.open(image_2)
        except:
            return 'wrong path'
        content = [img_1, img_2, prompt]
        working = True
        while working:
            try:
                responses = self.model.generate_content(
                    content,
                    stream=False,
                )
                self.all_keys[self.current_key]['last_inf_time'] = time.time()
                self.all_keys[self.current_key]['count_minute'] += 1
                self.all_keys[self.current_key]['count_day'] += 1
                break
            except Exception as e:
                print(e)
                working = self.load_key()
        if not working:
            return 'quotas'
        try:
            for response in responses:
                return response.text
        except Exception as e:
            print(e)
            return 'Gemini Error'

    def inference_one(self, image, prompt):
        try:
            img_1 = PIL_Image.open(image)
        except:
            return 'wrong path'
        content = [img_1, prompt]
        working = True
        while working:
            try:
                responses = self.model.generate_content(
                    content,
                    stream=False,
                )
                self.all_keys[self.current_key]['last_inf_time'] = time.time()
                self.all_keys[self.current_key]['count_minute'] += 1
                self.all_keys[self.current_key]['count_day'] += 1
                break
            except Exception as e:
                print(e)
                working = self.load_key()
        if not working:
            return 'quotas'
        try:
            for response in responses:
                return response.text
        except Exception as e:
            print(e)
            return 'Gemini Error'

class GPT():
    def __init__(self):
        api_key = 'YOUR API KEYS'
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def inference_two(self, image_1, image_2, prompt):
        try:
            en_1 = encode_image(image_1)
            en_2 = encode_image(image_2)
        except:
            return 'wrong path'
        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{en_1}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{en_2}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
        try:
            answer = response.json()['choices'][0]['message']['content']
        except:
            answer = 'GPT error'#response.json()
        return answer

    def inference_one(self, image, prompt):
        try:
            en = encode_image(image)
        except:
            return 'wrong path'
        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{en}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 3000
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
            answer = response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            answer = 'GPT error'
        return answer

    def inference_text(self, prompt):

        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 4000
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=payload)
            answer = response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            answer = "GPT Error"
        return answer


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')






def compute_clip_similarity(image1, image2, model, preprocess):
    image1_preprocess = preprocess(Image.open(image1)).unsqueeze(0).to('cuda')
    image1_features = model.encode_image(image1_preprocess)

    image2_preprocess = preprocess(Image.open(image2)).unsqueeze(0).to('cuda')
    image2_features = model.encode_image(image2_preprocess)

    cos = torch.nn.CosineSimilarity(dim=0)
    similarity = cos(image1_features[0], image2_features[0]).item()
    return similarity










