# Near deduplication

高效的单文件去重工具

### Setup

````
pip install -r requirements.txt
````

如果需要从huggingface上拉取文件，需要先登录:

````
huggingface-cli login
````

确保已经事先安装了git-lfs.

### Usage
#### 基本流程和理念
基本流程就是先远程拉取huggingface数据集，然后内部去重转化成本地数据集。然后多个本地数据集聚合去重，产生最终数据集

理念就是一个脚本反复跑，而不是开发很多个脚本，这才是好工具。
#### 处理远程数据集
```bash
python minhash_deduplication.py --dataset codeparrot/codeparrot-clean-valid \
    --split train \
    --column content \
    --cache-dir .cache \
    --min-ngram-size 5
```
#### 处理本地数据集
多个local-dataset合并去重
```bash
python minhash_deduplication.py \
    --local-dataset arxiv/deduplicated/ \
    --local-dataset NIHExPorter_output/deduplicated/ \
    --local-dataset PhilPapers_output/deduplicated/ \
    --local-dataset pes2o_train/deduplicated/ \
    --column text \
    --cache-dir rwkv_cache \
    --output RWKV-paper
```

### Python Implementation Analysis
为了了解当前去重实现的限制，重要的是要了解管道中的每个步骤如何影响总体时间：

哈希处理速度快，但对于长文档需要更长时间。哈希处理与核心数量和单核性能成比例。在使用 datasets 的缓存时，它也不需要太多内存。实现中在对文本分词后，会保留前10000个字词计算哈希值。
索引操作基本上是将哈希签名放入不同的桶中。这是管道中的一个瓶颈。为了节省内存，用时间换空间，每次只处理一个哈希桶。这样可以使得单机能处理的文本量提高10倍。
根据您决定如何分组重复项，您可以构建一个图，然后进行连接组件分析，这里使用了并查集这样的简单算法。
如何处理一组重复项也是一个开放性问题。在这种情况下，我们选择在组/集群内保留一个文档。

### 内存分析
最重要的是峰值内存分析，这决定了在什么硬件上能处理多大规模的数据。
代码中占用内存的主要是两个object，hashtable和union find。其他计算都可以被datasets缓存在硬盘。
实现中，hashtable是一个defaultdict(set)，uf的核心是一个parent dict，峰值内存估计如下：
| 数据量   | hashtable |   uf   | 总内存(MB) | 总内存(GB) |
| -------- | --------- | ------ | ---------- | ---------- |
| 100万    |     285   |   97   |     382    |    0.382   |
| 1000万   |    2775   |  895   |    3670    |    3.67    |
| 1亿      |   11000   | 29768  |   40768    |   40.768   |
