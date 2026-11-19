"""Exporta os HubParser completos para ONNX (fp32 e fp16) no padrão de pastas do transformers.js.

Entrada: input_ids [1, L] int64 (uma frase, sem padding, então sem máscara)
Saídas (linear):   upos [1,L,16], head [1,L,200], deprel [1,L,44], lemma [1,L,252], feats [1,L,174]
Saídas (biaffine): upos [1,L,16], arc [1,L,L] (dependente × cabeça), rel [1,L,L,44], lemma, feats

uso: python export_onnx.py HubParser-Base-B [...]   (sem argumentos: os quatro modelos)

A int8 dinâmica (onnxruntime.quantization) foi testada e descartada: perde de 3 a 32 pontos de LAS.
"""
import sys

import onnx
import torch
from onnxconverter_common import float16
from transformers import AutoModel

from comum import MODELOS, SAIDA, pasta_modelo


def biaffine_mm(b, x, y):
    """Mesmo resultado de Biaffine.forward, com matmul em vez de einsum: [B, O, Lx, Ly]."""
    if b.bias_x:
        x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
    if b.bias_y:
        y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], dim=-1)
    xw = torch.matmul(x.unsqueeze(1), b.weight.unsqueeze(0))          # [B, O, L, H]
    return torch.matmul(xw, y.unsqueeze(1).transpose(-1, -2))          # [B, O, L, L]


class Wrapper(torch.nn.Module):
    def __init__(self, m, float_out=False):
        super().__init__()
        self.m = m
        self.float_out = float_out
        self.biaffine = m.config.hubparser_head == "biaffine"

    def forward(self, input_ids):
        out = self._forward(input_ids)
        return tuple(o.float() for o in out) if self.float_out else out

    def _forward(self, input_ids):
        m = self.m
        pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        seq = m.bert(input_ids=input_ids, token_type_ids=torch.zeros_like(input_ids), position_ids=pos)[0]
        lemma, feats = m.lemma_classifier(seq), m.feats_classifier(seq)
        if not self.biaffine:
            return m.upos_classifier(seq), m.head_classifier(seq), m.deprel_classifier(seq), lemma, feats
        arc = biaffine_mm(m.arc_biaffine, m.arc_dep_mlp(seq), m.arc_head_mlp(seq))[:, 0]
        rel = biaffine_mm(m.rel_biaffine, m.rel_dep_mlp(seq), m.rel_head_mlp(seq)).permute(0, 2, 3, 1)
        return m.upos_classifier(seq), arc, rel, lemma, feats


def export(name):
    model = AutoModel.from_pretrained(pasta_modelo(name), trust_remote_code=True, attn_implementation="eager").eval()
    w = Wrapper(model).eval()
    out = SAIDA / name / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    names = (["upos", "arc", "rel", "lemma", "feats"] if w.biaffine
             else ["upos", "head", "deprel", "lemma", "feats"])
    ids = torch.tensor([[101, 1000, 2000, 3000, 102]])
    dyn = {"input_ids": {1: torch.export.Dim("L", min=3, max=512)}}

    def run(module, path):
        with torch.no_grad():
            torch.onnx.export(module, (ids,), str(path), input_names=["input_ids"], output_names=names,
                              dynamic_shapes=dyn, opset_version=18, dynamo=True, external_data=False, optimize=True)

    f32 = out / "model.onnx"
    run(w, f32)
    # fp16: converter o grafo fp32 já otimizado (exportar o modelo em .half() fixa o comprimento)
    onnx.save(float16.convert_float_to_float16(onnx.load(str(f32)), keep_io_types=True), str(out / "model_fp16.onnx"))
    for f in sorted(out.iterdir()):
        print(f"{name}: {f.name} {f.stat().st_size / 2**20:.0f} MB", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    for n in sys.argv[1:] or MODELOS:
        export(n)
