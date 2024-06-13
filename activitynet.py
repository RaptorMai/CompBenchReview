
# install fiftyone
# pip install fiftyone

import fiftyone
import sys

fiftyone.config.dataset_zoo_dir = "/local/scratch_2/jihyung/comp_imgs/dataset/"
dataset = fiftyone.zoo.load_zoo_dataset("activitynet-200", split="validation")
# dataset = fiftyone.zoo.load_zoo_dataset(
#         name="activitynet-200",
#         split="validation",
#         classes=["Horseback riding"],
#         max_duration=30,
#         max_samples=3,
# )