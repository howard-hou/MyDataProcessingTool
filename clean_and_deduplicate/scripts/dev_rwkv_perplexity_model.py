from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import os
import math
import torch

# set these before import RWKV
os.environ["RWKV_JIT_ON"] = "1"

world_model_01B = "model/RWKV-4-World-0.1B-v1-20230520-ctx4096.pth"
world_model_04B = "model/RWKV-4-World-0.4B-v1-20230529-ctx4096.pth"
# download models: https://huggingface.co/BlinkDL
model = RWKV(
    model=world_model_04B,
    strategy="cpu fp32",
)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

ctx = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
ctx = "\n蓄电池是将化学能直接转化成电能的一种装置，是按可再充电设计的电池"
print(ctx, end="")


def my_print(s):
    print(s, end="", flush=True)


import time

start = time.time()
tokens = pipeline.encode(ctx)
print("tokens", len(tokens))
out, state = model.forward(tokens[:1], None)
total_log_likelihood = 0
for i in range(1, len(tokens)):
    out, state = model.forward(tokens[i : i + 1], state)
    log_likelihood = out[tokens[i]].cpu().item()
    total_log_likelihood += log_likelihood

perplexity = math.exp(-total_log_likelihood / len(tokens))
# compute perplexity
print("perplexity", perplexity)
print("time", time.time() - start)
