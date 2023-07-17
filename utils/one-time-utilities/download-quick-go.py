import requests
import toml
import pickle
from tqdm import tqdm

config_dict = toml.load("config.toml")

class QuickGO:
    aspects = config_dict['gene-ontology']['NAMESPACES']
    def __init__(self, names, subgraph):
        self.subgraph = subgraph
        self.aspect = self.aspects[subgraph]
        self.names = names
        self.chunk = 100
    def get(self):
        for names in tqdm(zip(*(iter(self.names),) * self.chunk), desc = "Getting labels", total = len(self.names)//self.chunk):
            r = requests.get(self._get_urls(names), headers={"accept":"text/tsv"})
            content = r.content.decode().rstrip()
            parsed = self.parse(names, content)
            with open(f"train_terms_quickGO{self.subgraph}.tsv", "a") as f:
                f.writelines("".join(parsed))
    def parse(self, names, content):
        rv = []
        lines = content.split('\n')[1:]
        for name in names:
            relevant_lines = [line for line in lines if name in line]
            if len(relevant_lines) == 0:
                continue
            go_ids = set([int(line.split('\t')[4][3:]) for line in relevant_lines])
            rv.extend([f"{name}\tGO:{go_id:07d}\t{self.subgraph}O\n" for go_id in go_ids])
        return rv
    def _get_urls(self, names):
        names = ','.join([f"UniProtKB%3A{name}" for name in names])
        asp = self.aspect
        return f"https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?geneProductId={names}&geneProductType=protein&aspect={asp}"

def main():
    with open(config_dict['files']['EMBEDS_TEST'], 'rb') as f:
        names = list(pickle.load(f).keys())
    quick = QuickGO(names, 'BP').get()
    quick = QuickGO(names, 'CC').get()
    quick = QuickGO(names, 'MF').get()
if __name__ == "__main__":
    main()

