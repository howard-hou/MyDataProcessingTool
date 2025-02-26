import argparse
import pickle
import queue
import time
from collections import defaultdict
from glob import glob
from multiprocessing import Process, Queue

from datasketch.lean_minhash import LeanMinHash
from more_itertools import divide


def _H(hs):
    return bytes(hs.byteswap().data)


def split_files(input_dir, n_proc):
    files = glob(f"{input_dir}/*")
    files = sorted(files)
    parts = divide(n_proc, files)
    return [list(p) for p in parts]


def get_hashes(files, doc_queues, r):
    for fp in files:
        with open(fp, "rb") as fin:
            for item in pickle.load(fin):
                key = f"{item['file_name']}@{item['doc_id']}"
                minhash = LeanMinHash(item["hash"])
                for i, doc_queue in enumerate(doc_queues):
                    H = _H(minhash.hashvalues[i * r : (i + 1) * r])
                    doc_queue.put((key, H))


def lsh(out_file, doc_queue, idx):
    lsh_dict = defaultdict(str)
    i = 0
    out_path = out_file.parent / f"{out_file.stem}-{idx}.txt"
    f = open(out_path, "w")
    while True:
        try:
            key, H = doc_queue.get(timeout=30)
            cand = lsh_dict.get(H, "None")
            if cand != "None":
                f.write(f"{key} :: {cand}\n")
            else:
                lsh_dict[H] = key

            i += 1
        except queue.Empty:
            break

    print(f"Documents of {out_path.name}: {i}")
    f.close()


def generate_pairs(args):
    # size of the queue was tuned for optimal perf and memory constraints.
    doc_queues = [Queue() for _ in range(args.bands)]
    files = split_files(args.input_dir, args.processes)

    processes = []
    for process_id in range(args.processes):
        p = Process(
            target=get_hashes,
            args=(
                files[process_id],
                doc_queues,
                args.range,
            ),
        )
        processes.append(p)
        p.start()

    for process_id in range(args.bands):
        p = Process(
            target=lsh,
            args=(
                args.out_file,
                doc_queues[process_id],
                process_id,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--out_file")
    parser.add_argument(
        "--range",
        type=int,
        default=13,
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--processes",
        type=int,
    )
    args = parser.parse_args()

    generate_pairs(args)
