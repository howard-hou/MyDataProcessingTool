import os
import json
import random
import sys
from tqdm import tqdm

random.seed(22)

directory = sys.argv[1]  # Specify the directory path

# Get a list of file names in the directory
file_names = os.listdir(directory)

# Shuffle the file names randomly
file_names = sorted(file_names)

# Initialize a list to store sampled lines
sampled_lines = []


def sample_text(text, num_chars=100):
    # 5%开始100字 ... 50%开始100字 ... 95%开始100字
    text_len = len(text)
    start_index = text_len // 20
    mid_index = text_len // 2
    end_index = 19 * text_len // 20
    return dict(
        start=text[start_index : start_index + num_chars],
        middle=text[mid_index : mid_index + num_chars],
        end=text[end_index : end_index + num_chars],
    )


for file_name in tqdm(file_names):
    file_path = os.path.join(directory, file_name)
    # sample one line from top 100 lines
    rand_index = random.randrange(100)
    for i, line in enumerate(open(file_path, "r")):
        if i == rand_index:
            line = json.loads(line)
            sample = sample_text(line["text"])
            sample["file_name"] = file_name
            sample["line_index"] = rand_index
            sample["annotation"] = 1
            sample["comment"] = ""
            sampled_lines.append(sample)
            break


# Print the sampled lines for review
for i, line in enumerate(sampled_lines):
    if i == 0:
        print("--------  周福星 ----------")
    if i == 200:
        print("--------  尘 ----------")
    if i == 400:
        print("--------  霍华德 ----------")
    line = json.dumps(line, indent=4, ensure_ascii=False)
    print(line)
