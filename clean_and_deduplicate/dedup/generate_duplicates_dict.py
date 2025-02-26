import pickle
from collections import defaultdict


def generate_duplicates(args):
    print("Processing duplicates!!!")
    # load pickled components and other artifacts
    with open(args.input_file, "rb") as fin:
        components, n_components, reversed_mapper = pickle.load(fin)

    duplicates = defaultdict(set)
    n_duplicate_docs = 0
    for component in components:
        for j in range(1, len(component)):
            doc = reversed_mapper[component[j]]
            file_name, doc_idx = doc.split("@")
            duplicates[file_name].add(int(doc_idx))
            n_duplicate_docs += 1

    print("number of duplicate documents that will be removed:", n_duplicate_docs)
    # pickle
    with open(args.output_file, "wb") as fout:
        pickle.dump(duplicates, fout)
