conda create -n vl_eval python=3.10
conda activate vl_eval

pip install --upgrade openai
pip install -q -U google-generativeai
pip install ftfy regex tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/openai/CLIP.git