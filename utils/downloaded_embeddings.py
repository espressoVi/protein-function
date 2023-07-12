import numpy as np
import pickle
import torch
import sys

def change_format(embeds_file, ids_file):
    embeds = np.load(embeds_file)
    ids = np.load(ids_file)
    embeds_dict = {str(idx):embed for idx, embed in zip(ids, embeds)}
    return embeds_dict

def main():
    embeddings_file = sys.argv[1]
    ids_file = sys.argv[2]
    output_file = sys.argv[3]
    embeds_dict = change_format(embeddings_file, ids_file)
    with open(output_file, 'wb') as f:
        pickle.dump(embeds_dict, f)

if __name__ == "__main__":
    main()
