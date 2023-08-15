from GO import GeneOntology
import toml

config = toml.load("config.toml")

def test_names():
    with open(config['files']['TEST_SEQ'], 'r') as f:
        names = [i.split()[0][1:] for i in f.readlines() if ">" in i]
    return set(names)

def main():
    subgraph = "BP"
    GO = GeneOntology(subgraph)
    labels = GO.labels
    train_proteins = set([i for i,_ in labels])
    test_proteins = test_names()
    relevant = test_proteins - train_proteins
    with open(f"{subgraph}_list.tsv", "w") as f:
        f.writelines("\n".join(relevant))

if __name__ == "__main__":
    main()
