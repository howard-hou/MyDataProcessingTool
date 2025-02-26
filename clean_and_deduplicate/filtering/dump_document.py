import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import json
import pandas as pd

from tqdm import tqdm
import zstandard as zstd
from multiprocessing import Pool


def write_good_documents_to_zstd(
    input_file, output_file, keep_documents=None, file_duplicates=None
):
    """use zstandard to compress the file"""
    # if keep_documents is None: keep all documents

    # open a zstd compressed file for writing
    if file_duplicates is None:
        file_duplicates = set()

    num_docs = 0
    # output is .jsonl.zst file
    if not str(output_file).endswith(".jsonl"):
        output_file = str(output_file) + ".jsonl"
    if not str(output_file).endswith(".zst"):
        output_file = str(output_file) + ".zst"

    fout = zstd.open(output_file, "wb")
    for doc_id, line in enumerate(open(input_file)):
        if doc_id not in file_duplicates:
            if keep_documents is not None:
                if doc_id in keep_documents:
                    fout.write(line.encode("utf-8"))
                    num_docs += 1
            else:
                fout.write(line.encode("utf-8"))
                num_docs += 1
    fout.flush(zstd.FLUSH_FRAME)
    fout.close()
    return num_docs


def load_duplicates(duplicate_file):
    # pickle load
    with open(duplicate_file, "rb") as f:
        duplicates = pickle.load(f)
    return duplicates


def load_keep_documents(keep_file):
    df = pd.read_csv(keep_file, dtype={"file_name": str, "doc_id": int})
    keep_documents = df.groupby("file_name")["doc_id"].apply(set).to_dict()
    return keep_documents


def dump_good_documents(args):
    dedup_dir = Path(args.dedup_dir)
    duplicates = load_duplicates(args.duplicate_file)
    if Path(args.keep_file).exists():
        keep_documents = load_keep_documents(args.keep_file)
    else:
        # if keep_file not exists, keep all documents
        keep_documents = {}
    norm_dir = Path(args.input_dir)
    files = sorted(list(norm_dir.iterdir()))
    num_docs = 0
    # multiprocessing to write to zstd
    n_proc = min(len(files), args.n_proc)
    async_results = []
    with Pool(n_proc) as p:
        for file in files:
            async_results.append(
                p.apply_async(
                    write_good_documents_to_zstd,
                    args=(
                        file,
                        dedup_dir / file.name,
                        keep_documents.get(file.name, None),
                        duplicates.get(file.name, None),
                    ),
                )
            )
        for async_result in tqdm(async_results):
            num_docs += async_result.get()

    print("number of documents in the deduplicated dataset:", num_docs)
    print(f"Deduplicate Dataset Done and output to {dedup_dir}")
