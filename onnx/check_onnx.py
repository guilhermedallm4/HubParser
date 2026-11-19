"""Compara as saídas do ONNX (fp32 e fp16) com o PyTorch em frases aleatórias de vários comprimentos.

uso: python check_onnx.py HubParser-Base-B
"""
import sys

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModel

from comum import SAIDA, pasta_modelo
from export_onnx import Wrapper

name = sys.argv[1]
m = AutoModel.from_pretrained(pasta_modelo(name), trust_remote_code=True, attn_implementation="eager").eval()
w = Wrapper(m).eval()
for var in ["model.onnx", "model_fp16.onnx"]:
    s = ort.InferenceSession(str(SAIDA / name / "onnx" / var), providers=["CPUExecutionProvider"])
    for L in (3, 5, 17, 64, 512):
        ids = torch.randint(1000, 29000, (1, L)); ids[0, 0] = 101; ids[0, -1] = 102
        with torch.no_grad():
            ref = [t.numpy() for t in w(ids)]
        out = s.run(None, {"input_ids": ids.numpy()})
        diffs = [float(np.abs(a - b).max()) for a, b in zip(ref, out)]
        agree = [float((a.argmax(-1) == b.argmax(-1)).mean()) for a, b in zip(ref, out)]
        print(var, L, "maxdiff", [f"{d:.3g}" for d in diffs], "argmax-agree", [f"{a:.3f}" for a in agree], flush=True)
