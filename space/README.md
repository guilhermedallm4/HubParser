---
title: HubParser
colorFrom: purple
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
thumbnail: https://huggingface.co/spaces/guilhermedallm4/HubParser/resolve/main/thumbnail.png
short_description: Análise sintática UD do português, direto no navegador
models:
- guilhermedallm4/HubParser-Base-B
- guilhermedallm4/HubParser-Large-B
- guilhermedallm4/HubParser-Base-L
- guilhermedallm4/HubParser-Large-L
tags:
- dependency-parsing
- universal-dependencies
- portuguese
- webgpu
- onnx
---

# HubParser · H2IA UFPel

Interface web do **HubParser**, parser completo de Universal Dependencies para o português (UPOS, LEMMA,
FEATS, HEAD e DEPREL), treinado no Porttinari. Projeto do Hub de Inovação em Inteligência Artificial (H2IA)
da Universidade Federal de Pelotas.

A análise roda **inteiramente no navegador**, com o [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/):
na placa de vídeo (WebGPU) quando ela suporta meia precisão, ou no processador (WebAssembly) nos demais casos.
O texto não sai do computador de quem usa. Na primeira análise, o navegador baixa o modelo escolhido
(cerca de 210 MB no Base e 640 MB no Large), e ele fica guardado para as próximas visitas.

- Segmentação de frases e palavras no padrão UD, com as contrações desfeitas como no Porttinari ("da" → "de a").
- Árvore de dependências, tabela com lema e traços, e exportação em CoNLL-U e SVG.
- Decodificação Eisner (projetiva), MST/Chu-Liu-Edmonds ou gulosa.

Os modelos ficam em [guilhermedallm4/HubParser-Base-B](https://huggingface.co/guilhermedallm4/HubParser-Base-B) e
nos repositórios irmãos (pasta `onnx/`). No teste completo do Porttinari, esta versão no navegador reproduz
o modelo em PyTorch com diferença de no máximo 0,009 ponto percentual.
