"""Build the HubParser datasets used by the multilingual experiments.

  pt: Porttinari exactly as used in the dissertation (data/porttinari_dissertacao.jsonl.gz,
      derived from Porttinari-base, CC BY 4.0)
  en: UD_English-EWT r2.18 (CC BY-SA 4.0)
  es: UD_Spanish-AnCora r2.18 (CC BY 4.0)

Outputs (in data/built/): hubparser_pt, hubparser_en, hubparser_es and
hubparser_multilingual (pt+en+es concatenated), as HF datasets with columns
lang, tokens, upos, deprel, head_tags, upos_tags, deprel_tags.
Syntactic words only: multiword-token ranges and empty nodes are skipped.

Label ids come from data/labels.json (Porttinari order first, so Porttinari ids are
the dissertation ones; labels new to EN/ES are appended).
"""
import gzip
import json
import subprocess
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BUILT = DATA / "built"
UD_TAG = "r2.18"
UD = {"en": ("UD_English-EWT", "en_ewt"), "es": ("UD_Spanish-AnCora", "es_ancora")}
SPLITS = {"train": "train", "val": "dev", "test": "test"}


def read_conllu(path):
    sents, cur = [], {"tokens": [], "upos": [], "deprel": [], "head_tags": []}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur["tokens"]:
                sents.append(cur)
            cur = {"tokens": [], "upos": [], "deprel": [], "head_tags": []}
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        cur["tokens"].append(cols[1])
        cur["upos"].append(cols[3])
        cur["head_tags"].append(int(cols[6]))
        cur["deprel"].append(cols[7])
    if cur["tokens"]:
        sents.append(cur)
    return sents


def main():
    labels = json.loads((DATA / "labels.json").read_text())
    u2i = {u: i for i, u in enumerate(labels["upos"])}
    d2i = {d: i for i, d in enumerate(labels["deprel"])}

    raw = {"pt": {s: [] for s in SPLITS}}
    with gzip.open(DATA / "porttinari_dissertacao.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            raw["pt"][r.pop("split")].append(r)

    for lang, (repo, prefix) in UD.items():
        dst = DATA / "ud" / repo
        if not dst.exists():
            subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "-q", "--depth", "1", "--branch", UD_TAG,
                            f"https://github.com/UniversalDependencies/{repo}.git", str(dst)], check=True)
        raw[lang] = {s: read_conllu(dst / f"{prefix}-ud-{f}.conllu") for s, f in SPLITS.items()}

    per_lang = {}
    for lang, splits in raw.items():
        per_lang[lang] = DatasetDict({
            s: Dataset.from_list([dict(lang=lang, **r, upos_tags=[u2i[u] for u in r["upos"]],
                                       deprel_tags=[d2i[d] for d in r["deprel"]]) for r in rows])
            for s, rows in splits.items()})
        per_lang[lang].save_to_disk(str(BUILT / f"hubparser_{lang}"))
        print(lang, {s: len(d) for s, d in per_lang[lang].items()})
    joint = DatasetDict({s: concatenate_datasets([per_lang[l][s] for l in ("pt", "en", "es")]) for s in SPLITS})
    joint.save_to_disk(str(BUILT / "hubparser_multilingual"))
    print("multilingual", {s: len(d) for s, d in joint.items()})


if __name__ == "__main__":
    main()
