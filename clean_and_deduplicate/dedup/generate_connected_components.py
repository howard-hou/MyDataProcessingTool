import argparse
import pickle
import time
from glob import glob

import networkit as nk
from tqdm import tqdm


def construct_graph(set_of_duplicate_pairs):
    G = nk.Graph()
    mapper = {}
    for pair in tqdm(set_of_duplicate_pairs, desc="Constructing graph"):
        node1_name, node2_name = pair
        if node1_name not in mapper:
            mapper[node1_name] = G.addNode()
        if node2_name not in mapper:
            mapper[node2_name] = G.addNode()
        G.addEdge(mapper[node1_name], mapper[node2_name])
    return G, mapper


def find_connected_components(G):
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    return cc.getComponents(), cc.numberOfComponents()


def generate_connected_components_mp(args):
    files = glob(f"{args.input_dir}/*.txt")

    print("Started graph building")
    # load pickled duplicate pairs
    set_of_duplicate_pairs = set()
    for fp in files:
        with open(fp, "r") as f:
            for line in f:
                pair = tuple(line.strip().split(" :: "))
                if pair[0] != pair[1]:
                    set_of_duplicate_pairs.add(pair)
    print(
        "length of the set of duplicates:",
        len(set_of_duplicate_pairs),
    )

    # generate a graph using id's as nodes and a pair of ids as an edge
    nk.setNumberOfThreads(60)
    G, mapper = construct_graph(set_of_duplicate_pairs)
    components, n_components = find_connected_components(G)
    print("number of connected components:", n_components)

    reversed_mapper = {value: key for key, value in mapper.items()}

    # dump pickled cc on disk and load if needed
    with open(args.out_file, "wb") as fout:
        pickle.dump((components, n_components, reversed_mapper), fout)
    print("Graph generated duplicates list!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--out_file")
    args = parser.parse_args()
    generate_connected_components_mp(args)
