import argparse
import os
import pickle
import queue
import re
import string
import sys
import time
import json
from collections import Counter, defaultdict
from glob import glob
from multiprocessing import Pool, Process, Queue, cpu_count
from tqdm import tqdm
import math

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def clean(s):
    # lower cased
    s = s.lower()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    return s


def count_document(file, topk=1000):
    """calculate count for each word in the file
    Args:  file: file path
           topk: top k words to return
    """
    word_count = Counter()
    document_count = 0
    with open(file) as f:
        for line in f:
            sentences = json.loads(line)["text"].split("\n")
            document_count += 1
            words = []
            for sentence in sentences:
                words.extend([word for word in sentence.split() if word])
            word_count.update(list(set(words)))
    return word_count.most_common(topk), document_count


def calculate_idf(args):
    """calculate idf for each word in the input directory"""
    files = sorted(glob(f"{args.input_dir}/*"))
    # calculate word count and document count by multiprocessing
    with Pool(args.n_proc, maxtasksperchild=args.maxtasksperchild) as p:
        async_results = [
            p.apply_async(count_document, args=(file, args.topk)) for file in files
        ]
        results = [
            async_result.get()
            for async_result in tqdm(async_results, desc="Calculate idf")
        ]
    #
    total_document_count = 0
    total_word_count = defaultdict(int)
    for word_count, document_count in results:
        total_document_count += document_count
        for word, count in word_count:
            total_word_count[word] += count
    # calculate idf
    idf_scores = {}
    for word, count in total_word_count.items():
        # use 1 + log(N / df) as idf
        idf_scores[word] = 1 + math.log(total_document_count / count)
    # sort by idf
    idf_scores = sorted(idf_scores.items(), key=lambda x: x[1])
    with open(args.output_file, "w") as fout:
        for word, idf in idf_scores:
            fout.write(f"{word}\t{idf}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Dataset input directory.")
    parser.add_argument("output_file", help="File to output short docs to.")
    args = parser.parse_args()
    calculate_idf(args)
