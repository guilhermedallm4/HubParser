"""Shared pieces of the multilingual HubParser experiments: data, labels, model
construction, tokenization/label alignment, collator, metrics and evaluation."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoConfig

from .modeling_hubparser import hubparser_class

ML_ROOT = Path(__file__).resolve().parent.parent           # multilingual/
DATA_DIR = Path(os.environ.get("HUBPARSER_ML_DATA", ML_ROOT / "data" / "built"))
LABELS = json.loads((ML_ROOT / "data" / "labels.json").read_text())
NUM_HEAD_LABELS = 200          # linear head: absolute word positions 0..199 (max sentence = 159 words)

ENCODERS = {
    "bert": "google-bert/bert-base-cased",
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "bertimbau-base": "neuralmind/bert-base-portuguese-cased",
    "bertimbau-large": "neuralmind/bert-large-portuguese-cased",
    "jabuticabert": "amadeusai/modernJabuticaBERT-Base-1k",
}
# head variants: linear | biaffine (dissertation, word-index target) | biaffine_fix (first-subtoken target)
HEADS = ("linear", "biaffine", "biaffine_fix")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_corpus(corpus):
    """corpus = "multilingual" (pt+en+es) or a single language "pt" | "en" | "es"."""
    return load_from_disk(str(DATA_DIR / f"hubparser_{corpus}"))


def labels_for(corpus):
    """Porttinari-only runs keep the dissertation inventory (16 UPOS, 44 DEPREL)."""
    if corpus == "pt":
        return LABELS["pt_upos"], LABELS["pt_deprel"]
    return LABELS["upos"], LABELS["deprel"]


def build_model(encoder, head, corpus):
    base = ENCODERS[encoder]
    config = AutoConfig.from_pretrained(base)
    config.upos_labels, config.deprel_labels = labels_for(corpus)
    config.hubparser_head = "linear" if head == "linear" else "biaffine"
    config.head_target = "first_subtoken" if head == "biaffine_fix" else "word_index"
    if head == "linear":
        config.num_head_labels = NUM_HEAD_LABELS
    else:
        config.arc_hidden, config.rel_hidden, config.mlp_dropout = 500, 100, 0.33
    cls = hubparser_class(config.hubparser_head, config.model_type)
    model, info = cls.from_pretrained(base, config=config, output_loading_info=True)
    enc_missing = [k for k in info["missing_keys"] if k.split(".")[0] in ("bert", "model") and "position_ids" not in k and ".pooler." not in k]
    assert not enc_missing, f"encoder weights not loaded: {enc_missing[:5]}"
    return model, cls


def tokenize(dataset, tokenizer, head, pad_to_512):
    """Every subtoken of a word gets the word's labels (as in the dissertation).
    biaffine_fix: the head label is the first-subtoken position of the head word
    (root -> 0 = [CLS]); otherwise it is the head's word index."""
    def fn(examples):
        enc = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=512,
                        padding="max_length" if pad_to_512 else False)
        out = {k: [] for k in ("deprel_label", "upos_label", "head_label")}
        for i in range(len(examples["tokens"])):
            wid = enc.word_ids(i)
            first = {}
            for p, w in enumerate(wid):
                if w is not None and w not in first:
                    first[w] = p
            heads = examples["head_tags"][i]
            if head == "biaffine_fix":
                heads = [0 if h == 0 else first.get(h - 1, -100) for h in heads]
            for src, dst in (("deprel_tags", "deprel_label"), ("upos_tags", "upos_label")):
                lab = examples[src][i]
                out[dst].append([-100 if w is None else lab[w] for w in wid])
            out["head_label"].append([-100 if w is None else heads[w] for w in wid])
        enc.update(out)
        return enc
    return dataset.map(fn, batched=True, remove_columns=dataset.column_names)


def make_collator(pad_id):
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {
            "input_ids": torch.tensor([b["input_ids"] + [pad_id] * (max_len - len(b["input_ids"])) for b in batch]),
            "attention_mask": torch.tensor([b["attention_mask"] + [0] * (max_len - len(b["attention_mask"])) for b in batch]),
        }
        if "token_type_ids" in batch[0]:
            out["token_type_ids"] = torch.tensor([b["token_type_ids"] + [0] * (max_len - len(b["token_type_ids"])) for b in batch])
        for k in ("deprel_label", "upos_label", "head_label"):
            out[k] = torch.tensor([b[k] + [-100] * (max_len - len(b[k])) for b in batch])
        return out
    return collate


def preprocess_logits_for_metrics(logits, labels):
    # argmax before accumulation: biaffine head logits are [B, L, L] and L varies per batch
    return tuple(l.argmax(-1) for l in logits[:3])


def compute_metrics(eval_pred):
    """Subtoken-level validation metrics, as in the dissertation's Trainer runs."""
    (deprel_preds, upos_preds, head_preds), (deprel_labels, upos_labels, head_labels) = eval_pred
    valid = (head_labels != -100) & (deprel_labels != -100)
    uas = (head_preds[valid] == head_labels[valid]).mean()
    las = ((head_preds[valid] == head_labels[valid]) & (deprel_preds[valid] == deprel_labels[valid])).mean()
    upos_mask = upos_labels != -100
    return {"uas": float(uas), "las": float(las),
            "upos_accuracy": float((upos_preds[upos_mask] == upos_labels[upos_mask]).mean())}


def evaluate_test(model, tokenizer, test, decoding):
    """Word-level UPOS/UAS/LAS per language (first subtoken; punctuation included)."""
    parsed = model.parse(test["tokens"], tokenizer, decoding=decoding)
    acc = {}
    for sent, lang, gu, gd, gh in zip(parsed, test["lang"], test["upos"], test["deprel"], test["head_tags"]):
        a = acc.setdefault(lang, dict(words=0, upos=0, uas=0, las=0))
        for w in sent:
            j = w["id"] - 1
            a["words"] += 1
            a["upos"] += w["upos"] == gu[j]
            if w["head"] == gh[j]:
                a["uas"] += 1
                a["las"] += w["deprel"] == gd[j]
    return {lang: {"upos": 100 * a["upos"] / a["words"], "uas": 100 * a["uas"] / a["words"],
                   "las": 100 * a["las"] / a["words"], "words": a["words"]} for lang, a in acc.items()}
