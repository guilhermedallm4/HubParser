"""Publica a aplicação no Space estático H2IA/HubParser.

uso (na raiz do repositório):  cd app && npm ci && npm run build && cd .. && python space/publicar.py
"""
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

RAIZ = Path(__file__).resolve().parent.parent
dist = RAIZ / "app" / "dist"
assert (dist / "index.html").exists(), "rode antes: cd app && npm run build"
with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(dist, tmp, dirs_exist_ok=True)
    for f in ("README.md", "thumbnail.png"):
        shutil.copy(RAIZ / "space" / f, tmp)
    c = HfApi().upload_folder(repo_id="H2IA/HubParser", repo_type="space", folder_path=tmp,
                              commit_message="Atualiza a aplicação", delete_patterns=["assets/*"])
    print(c.commit_url)
