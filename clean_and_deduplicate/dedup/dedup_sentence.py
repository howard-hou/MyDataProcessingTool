from datasketch import MinHash, MinHashLSH
import argparse
import json
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd


def deduplicate_one_file_lastk_sentence_idf(
    input_file, output_dir, word2idf, lastk=5, threshold=0.8
):
    ans = []
    for docid, line in enumerate(open(input_file)):
        data = json.loads(line)
        text = data["text"]
        sentences = text.split("\n")
        end = len(sentences)
        start = end - lastk if end > lastk else 0
        start_index = 0
        end_index = end
        for i in range(start, end):
            words = sentences[i].split()
            avg_idf = sum([word2idf.get(w, 20) for w in words]) / len(words)
            if avg_idf < threshold:
                end_index = i
                break
        if end_index != end:
            ans.append(f"{docid}@{start_index}@{end_index}")

    output_file = Path(output_dir) / input_file.name
    with open(output_file, "w") as f:
        f.write("\n".join(ans))


def create_minhash(text, num_perm=128):
    minhash = MinHash(num_perm=num_perm)
    for word in text.split():
        minhash.update(word.encode("utf8"))
    return minhash


def deduplicate_one_file_lastk_sentence(
    input_file, output_dir, lastk=5, threshold=0.8, num_perm=128
):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    ans = []
    for docid, line in enumerate(open(input_file)):
        data = json.loads(line)
        text = data["text"]
        sentences = text.split("\n")
        end = len(sentences)
        start = end - lastk if end > lastk else 0
        start_index = 0
        end_index = end
        for i in range(start, end):
            minhash = create_minhash(sentences[i])
            dedup_index = lsh.query(minhash)
            if dedup_index:
                end_index = i
                break
            lsh.insert(f"{docid}@{i}", minhash)
        if end_index != end:
            ans.append(f"{docid}@{start_index}@{end_index}")

    output_file = Path(output_dir) / input_file.name
    with open(output_file, "w") as f:
        f.write("\n".join(ans))


def deduplicate_lastk_sentence(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    input_files = [f for f in input_dir.glob("*")]
    if args.idf_file is not None:
        idf_df = pd.read_csv(args.idf_file, sep="\t", header=None)
        word2idf = dict(zip(idf_df[0], idf_df[1]))

    if args.n_proc == 1:
        # single process
        for input_file in tqdm(input_files, desc="Dedup sentence"):
            deduplicate_one_file_lastk_sentence_idf(
                input_file, output_dir, word2idf, args.lastk, args.threshold
            )
    else:
        # multiprocessing, apply async
        with Pool(args.n_proc, maxtasksperchild=args.maxtasksperchild) as p:
            async_results = []
            for input_file in input_files:
                async_result = p.apply_async(
                    deduplicate_one_file_lastk_sentence_idf,
                    args=(
                        input_file,
                        output_dir,
                        word2idf,
                        args.lastk,
                        args.threshold,
                    ),
                )
                async_results.append(async_result)
            for async_result in tqdm(async_results, desc="Dedup sentence"):
                async_result.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str, default="duplicate_sentences")
    parser.add_argument("--lastk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--num_perm", type=int, default=128)
    parser.add_argument("--n_proc", type=int, default=1)
    parser.add_argument("--maxtasksperchild", type=int, default=1)
    parser.add_argument("--idf_file", type=str, default=None)
    args = parser.parse_args()
    deduplicate_lastk_sentence(args)
