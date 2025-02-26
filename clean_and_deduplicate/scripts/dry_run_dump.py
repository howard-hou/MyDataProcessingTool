import pickle
import pandas as pd
import argparse


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
    duplicates = load_duplicates(args.duplicate_file)
    keep_documents = load_keep_documents(args.keep_file)
    # tke the union of the keys
    file_names = set(duplicates.keys()).union(set(keep_documents.keys()))
    file_names = sorted(list(file_names))
    num_docs = 0
    for file_name in file_names:
        keep = keep_documents.get(file_name, set())
        dup = duplicates.get(file_name, set())
        for i in range(10000):
            if i not in dup:
                if i in keep:
                    num_docs += 1
    print("num_docs in final dataset", num_docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("duplicate_file", help="Input directory with documents.")
    parser.add_argument("keep_file", help="Output directto.")
    args = parser.parse_args()

    dump_good_documents(args)
