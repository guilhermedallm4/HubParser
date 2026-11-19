"""Avalia um ONNX exportado num CoNLL-U de teste, com os decodificadores Python originais.

Reproduz HubParserMixin.parse + _FullExtra.parse a partir das saídas do ONNX. Grava as métricas e as
previsões em <saida>/<modelo>/ (as previsões servem para conferir a versão JavaScript, teste_js/validar.mjs).

uso: python eval_onnx.py HubParser-Base-B model_fp16 --gold test.conllu [--ref final.json] [--limite 100]

--ref: JSON com as métricas do PyTorch por decodificação ({"test": {"eisner": {"las": ...}}}) para calcular a diferença.
"""
import argparse
import json
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from comum import SAIDA, modulo_hubparser, pasta_modelo

mh = None


def read_conllu(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append(cur)
            cur = []
        elif not line.startswith("#"):
            c = line.split("\t")
            if c[0].isdigit():
                cur.append(dict(form=c[1], lemma=c[2], upos=c[3], feats=c[5], head=int(c[6]), deprel=c[7]))
    if cur:
        sents.append(cur)
    return sents


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def parse_one(sess, tok, cfg, words, decodings):
    enc = tok([words], is_split_into_words=True, truncation=True, max_length=512)
    ids = np.array([enc["input_ids"][0]], dtype=np.int64)
    word_ids = enc.word_ids(0)
    first = {}
    for p, w in enumerate(word_ids):
        if w is not None and w not in first:
            first[w] = p
    n = len(first)
    fp = [first[i] for i in range(n)]
    biaffine = cfg["hubparser_head"] == "biaffine"
    o = dict(zip([x.name for x in sess.get_outputs()], sess.run(None, {"input_ids": ids})))
    upos = [cfg["upos_labels"][i] for i in o["upos"][0][fp].argmax(-1)]
    lemma = [mh.apply_rule(words[d], cfg["lemma_rules"][r]) for d, r in enumerate(o["lemma"][0][fp].argmax(-1))]
    feats = [cfg["feats_labels"][i] for i in o["feats"][0][fp].argmax(-1)]
    if biaffine:
        cols = [0] + fp
        rows = o["arc"][0][fp].astype(np.float64)
        pos2word = {p: (w + 1 if w is not None else 0) for p, w in enumerate(word_ids)}
    else:
        cols = list(range(n + 1))
        rows = o["head"][0][fp].astype(np.float64)
    res = {}
    for dec in decodings:
        if dec == "greedy":
            pc = rows.argmax(-1).tolist()
            heads = [pos2word.get(c, 0) for c in pc] if biaffine else pc
        else:
            S = log_softmax(rows)[:, cols]
            heads = mh.eisner(S) if dec == "eisner" else mh.chu_liu_edmonds(S)
            pc = [cols[h] if h < len(cols) else h for h in heads]
        if biaffine:
            rel = o["rel"][0]
            dep = [int(rel[fp[d], min(c, rel.shape[1] - 1)].argmax()) for d, c in enumerate(pc)]
        else:
            dep = o["deprel"][0][fp].argmax(-1).tolist()
        res[dec] = dict(head=[int(h) for h in heads], deprel=[cfg["deprel_labels"][i] for i in dep])
    return dict(upos=upos, lemma=lemma, feats=feats, **res)


def main():
    global mh
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo")
    ap.add_argument("variante", help="model (fp32) ou model_fp16")
    ap.add_argument("--gold", required=True, help="CoNLL-U de teste (palavras sintáticas, sem linhas MWT)")
    ap.add_argument("--ref", help="JSON com as métricas do PyTorch")
    ap.add_argument("--limite", type=int)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    name, variant = a.modelo, a.variante
    mh = modulo_hubparser(name)
    cfg = json.load(open(pasta_modelo(name) / "config.json"))
    tok = AutoTokenizer.from_pretrained(pasta_modelo(name))
    gold = read_conllu(a.gold)[:a.limite]
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(str(SAIDA / name / "onnx" / f"{variant}.onnx"), so, providers=["CPUExecutionProvider"])
    decs = ["greedy", "eisner", "mst"]
    t0 = time.time()
    preds = [parse_one(sess, tok, cfg, [w["form"] for w in s], decs) for s in gold]
    secs = time.time() - t0
    tot = sum(len(s) for s in gold)
    met = {}
    for dec in decs:
        c = dict(upos=0, lemma=0, feats=0, uas=0, las=0)
        for s, p in zip(gold, preds):
            for j, g in enumerate(s):
                c["upos"] += p["upos"][j] == g["upos"]
                c["lemma"] += p["lemma"][j] == g["lemma"]
                c["feats"] += p["feats"][j] == g["feats"]
                hu = p[dec]["head"][j] == g["head"]
                c["uas"] += hu
                c["las"] += hu and p[dec]["deprel"][j] == g["deprel"]
        met[dec] = {"words": tot, **{k: 100 * v / tot for k, v in c.items()}}
    delta = None
    if a.ref:
        ref = json.load(open(a.ref))
        ref = ref.get("test", ref)
        delta = {d: {k: round(met[d][k] - ref[d][k], 3) for k in ("upos", "lemma", "feats", "uas", "las")} for d in decs}
    rep = {"model": name, "variant": variant, "sentences": len(gold), "seconds": round(secs, 1), "metrics": met,
           "delta_vs_pytorch": delta}
    (SAIDA / name / f"eval_{variant}.json").write_text(json.dumps(rep, indent=1))
    (SAIDA / name / f"pred_{variant}.json").write_text(json.dumps(preds, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("model", "variant", "sentences", "seconds", "metrics", "delta_vs_pytorch")}), flush=True)


if __name__ == "__main__":
    main()
