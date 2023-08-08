import requests
import toml
import pickle
from tqdm import tqdm
import json

config_dict = toml.load("config.toml")
evidence_codes = "ECO:0000269,ECO:0000314,ECO:0000353,ECO:0000315,ECO:0000316,ECO:0000270,ECO:0000304,ECO:0000305"

class QuickGO:
    aspects = config_dict['gene-ontology']['NAMESPACES']
    def __init__(self, names):
        self.names = names
        self.chunk = 100
        self.evidence = set(self._get_evidence())
    def get(self):
        for names in tqdm(zip(*(iter(self.names),) * self.chunk), desc = "Getting labels", total = len(self.names)//self.chunk):
            content = self.get_content(names)
            if content is None:
                continue
            parsed = self.parse(names, content)
            with open(f"train_terms_quickGO.tsv", "a") as f:
                f.writelines("".join(parsed))
    def get_content(self, names):
        try:
            r = requests.get(self._get_urls(names), headers={"accept":"text/tsv",}, timeout=10)
            content = r.content.decode().rstrip()
            return content
        except requests.exceptions.Timeout:
            with open("failed.tsv","a") as f:
                f.writelines("".join([f"{i}\n" for i in names]))
            return None
    def parse(self, names, contents):
        rv = []
        contents = contents.split('\n')[2:]
        for content in contents:
            line = content.split('\t')
            _name = line[1]
            _goId = line[4]
            _evidence = line[6]
            if _evidence not in self.evidence or _goId == "GO:0005515":
                continue
            rv.append(f"{_name}\tGO:{_goId}\n")
        return rv
    def _get_urls(self, names):
        names = '%2C'.join([f"{name}" for name in names])
        return f"https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?geneProductId={names}&geneProductType=protein"
    def is_relevant(self, line, name):
        if name not in line:
            return False
        for evidence in self.evidence:
            if evidence in line:
                return True
        return False
    def _get_evidence(self):
        with open("./utils/evidence.txt","r") as f:
            evidence = [i.rstrip() for i in f.readlines()]
        return evidence

def main():
    with open("./data/names.tsv","r") as f:
        names = [i.rstrip() for i in f.readlines()]
    names = names[200:]
    quick = QuickGO(names).get()

if __name__ == "__main__":
    main()
