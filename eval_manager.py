from abc import ABC, abstractmethod
from setup import compute_clip_similarity, GEMINI, GPT
import json

class EvalManager(ABC):
    def __init__(self, model):
        self.model = model

    def setup(self):
        if self.model == 'gpt':
            self.inf_model = GPT()
        elif self.model == 'gemini':
            self.inf_model = GEMINI()
        else:
            raise NotImplementedError

    def load_data(self, input_file):
        self.input_file = input_file
        with open(input_file, 'r') as file:
            try:
                self.data = json.load(file)
            except json.JSONDecodeError:
                print(f"Failed to decode JSON")

    @abstractmethod
    def eval(self, prompt, output_name):
        raise NotImplementedError

