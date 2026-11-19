"""Caminhos compartilhados pelos scripts de conversão e validação."""
import importlib.util
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DONO = "guilhermedallm4"
MODELOS = ["HubParser-Base-B", "HubParser-Base-L", "HubParser-Large-B", "HubParser-Large-L"]
# onde ficam os ONNX gerados: <SAIDA>/<modelo>/onnx/model*.onnx
SAIDA = Path(os.environ.get("HUBPARSER_ONNX", Path(__file__).resolve().parent / "saida"))


def pasta_modelo(nome: str) -> Path:
    """Pasta local do modelo PyTorch: HUBPARSER_MODELOS/<nome> se existir; senão baixa do Hub."""
    local = os.environ.get("HUBPARSER_MODELOS")
    if local and (Path(local) / nome / "config.json").exists():
        return Path(local) / nome
    return Path(snapshot_download(f"{DONO}/{nome}", allow_patterns=[
        "config.json", "tokenizer*.json", "model.safetensors", "modeling_hubparser.py"]))


def modulo_hubparser(nome: str):
    """Importa o modeling_hubparser.py do modelo (decodificadores Eisner/MST e regras de lema)."""
    spec = importlib.util.spec_from_file_location("modeling_hubparser", pasta_modelo(nome) / "modeling_hubparser.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
