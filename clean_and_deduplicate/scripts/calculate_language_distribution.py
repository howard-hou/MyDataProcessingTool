import pandas as pd
import matplotlib.pyplot as plt
import sys
import json
from pathlib import Path

input_dir = Path(sys.argv[1])
# 读取多个JSONL文件
file_paths = input_dir.glob("*")  # 匹配所有的JSONL文件
data = []
for file_path in file_paths:
    with open(file_path, "r") as file:
        for line in file:
            j = json.loads(line)
            j["language_label"] = j["language_label"].replace("__label__", "")
            data.append(j)

# 提取字段并创建DataFrame
df = pd.DataFrame(data)
field_values = df["language_label"]

# 统计字段值计数
value_counts = field_values.value_counts(normalize=True)
# print top 10
print(value_counts[:10])

# 绘制topk直方图
topk = 10
plt.bar(value_counts[:topk].index, value_counts[:topk].values)
plt.xticks(rotation=45)
plt.xlabel("Language")
plt.ylabel("Count")
plt.title(f"Top {topk} languages")

# save to file
plt.savefig("language_distribution.png")
