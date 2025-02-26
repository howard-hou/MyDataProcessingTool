import argparse
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def deduplicate_one_file_lastk_sentence_idf(
    input_file, output_file, word2idf, lastk=5, threshold=0.8
):
    w = open(output_file, "w")
    for line in tqdm(open(input_file)):
        data = json.loads(line)
        text = data["text"]
        sentences = text.split("\n")
        end = len(sentences)
        start = end - lastk if end > lastk else 0
        start_index = 0
        end_index = end
        for i in range(start, end):
            words = sentences[i].split()
            if len(words) == 0:
                continue
            avg_idf = sum([word2idf.get(w, 20) for w in words]) / len(words)
            if avg_idf < threshold:
                end_index = i
                break
        text_dedup = "\n".join(sentences[start_index:end_index])
        data["text"] = text_dedup
        w.write(json.dumps(data, ensure_ascii=False) + "\n")


def deduplicate_lastk_sentence(args):
    idf_df = pd.read_csv(args.idf_file, sep="\t", header=None)
    word2idf = dict(zip(idf_df[0], idf_df[1]))

    # single process
    deduplicate_one_file_lastk_sentence_idf(
        args.input_file, args.output_file, word2idf, args.lastk, args.threshold
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("idf_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--lastk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=20)
    args = parser.parse_args()
    deduplicate_lastk_sentence(args)
