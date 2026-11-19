"""Referências do Python para conferir o porte JavaScript (teste_js/validar.mjs):
ids do tokenizador do HF (is_split_into_words=True) e segmentação de textos livres.

uso: python dump_tok.py --gold test.conllu
"""
import argparse
import json

from transformers import AutoTokenizer

from comum import SAIDA, pasta_modelo
from eval_onnx import read_conllu
from segmentacao_ref import segmentar

ap = argparse.ArgumentParser()
ap.add_argument("--gold", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(pasta_modelo("HubParser-Base-B"))
sents = [[w["form"] for w in s] for s in read_conllu(a.gold)]
extra = [["Ｏ", "café", "naïve", "ﬁm", "x²", "😀", "中文字", "e-mail", "R$", "50,00", "–", "…", "a" * 120, "Ação", "ǅ", "İstanbul"]]
ids = [tok(s, is_split_into_words=True)["input_ids"] for s in sents + extra]
textos = [
    "O Sr. Silva chegou às 12h30. Ele disse: \"Vamos!\" Depois saiu... E voltou?",
    "Visite https://ia.ufpel.edu.br/pt ou escreva para contato@exemplo.com.br, por favor.",
    "Custa R$ 1.234,56 (aprox. 3,5% a mais). DA casa NA rua; Pela manhã, nele confiamos.",
    "Linha um sem ponto\nLinha dois. 2026 foi bom! «Citação» — fim.\r\nÀquela hora, daqui a pouco, fazê-lo-ia.",
    "Dr. João e a Profa. Ana vão ao cinema no dia 10 de jan. de 2027. Ótimo!",
]
segs = [[{"texto": f.texto, "palavras": f.palavras, "mwt": [{"inicio": i, "fim": j, "forma": c} for i, j, c in f.mwt]}
         for f in segmentar(t)] for t in textos]
SAIDA.mkdir(parents=True, exist_ok=True)
(SAIDA / "ref_tok.json").write_text(json.dumps({"sents": sents + extra, "ids": ids, "textos": textos, "segs": segs},
                                               ensure_ascii=False))
print(len(ids), "frases de referência")
