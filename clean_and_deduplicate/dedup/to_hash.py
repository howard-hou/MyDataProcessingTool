import argparse
import json
import os
import pickle
import re
import string
import sys
from multiprocessing import Pool
from pathlib import Path

from datasketch import MinHash
from nltk import ngrams
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def get_features(s, width, max_len=1000):
    # lower cased
    s = s.lower()
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    # split into words by whitespace and truncate to max_len
    s = s.split()[:max_len]
    return map(lambda x: " ".join(x), ngrams(s, width))


def get_files(input_dir):
    files = sorted(Path(input_dir).glob("*"))
    return files


def get_documents(input_file):
    with open(input_file) as rdr:
        for doc_id, doc in enumerate(rdr):
            yield doc, input_file, doc_id


def to_minhash(file, output_dir, ngram):
    buckets = []
    documents = get_documents(file)
    for doc, file_path, doc_id in documents:
        text = json.loads(doc)["text"]
        file_name = Path(file_path).name

        m = MinHash(num_perm=128)
        for x in get_features(text, ngram):
            m.update(x.encode("utf8"))
        buckets.append(
            {
                "file_name": file_name,
                "doc_id": doc_id,
                "hash": m,
            }
        )
    file_name = Path(file).name
    output_results(output_dir, file_name, buckets)


def output_results(output_dir, file_name, results):
    with open(f"{output_dir}/{file_name}.pickle", "wb") as fout:
        pickle.dump(results, fout)


def filter_exists_file(files, output_dir):
    # if output file exists, skip
    exists_filenames = [f.stem for f in output_dir.iterdir()]
    print(
        f"Generating hashes: Found {len(exists_filenames)} existing files, skip them."
    )
    new_files = []
    for file in files:
        file_name = Path(file).name
        if file_name not in exists_filenames:
            new_files.append(file)
    return new_files


def generate_hashes(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_files(args.input_dir)
    files = filter_exists_file(files, output_dir)
    with Pool(args.cpu_count, maxtasksperchild=1) as p:
        async_results = {}
        for file_ in files:
            file_name = Path(file_).name
            async_result = p.apply_async(to_minhash, (file_, output_dir, args.ngram))
            async_results[async_result] = file_name

        for async_result in tqdm(
            async_results, desc="Generating hashes", total=len(files)
        ):
            async_result.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Input directory with documents.")
    parser.add_argument(
        "output_dir", help="Output directory to output minhash files to."
    )
    parser.add_argument(
        "--ngram", type=int, default=13, help="The window size", required=False
    )
    parser.add_argument("--cpu_count", type=int, default=-1, required=False)
    args = parser.parse_args()
    generate_hashes(args)
