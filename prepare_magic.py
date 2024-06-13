import json
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
import os
from openai import OpenAI
import base64
import requests
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import http.client
import typing
import urllib.request

import IPython.display
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps

import torch
import clip
from PIL import Image

import argparse




def setup_parser():
    parser = argparse.ArgumentParser(description='magic_brush')
    parser.add_argument('--path', default='magic_brush/dev/')
