# HubParser — Parsing de Dependências Multi-tarefa para o Português Brasileiro

**HubParser** é o parser de dependências desenvolvido na dissertação de mestrado
(PPGC/UFPel, 2026). É um estudo próprio: modelos multi-tarefa (UPOS + DEPREL + HEAD)
com encoders BERT ajustados ponta a ponta e dois cabeçotes de predição (linear e
biaffine), treinados no corpus **Porttinari**. A dissertação também traz um estudo de
explicabilidade por camada (logit lens, probes, early exit, block skip).

O [PortParser](https://aclanthology.org/2024.propor-1.41/) (Lopes et al., 2024) é usado
**apenas como baseline** de comparação no mesmo split do Porttinari (notebook 10). O
HubParser não estende nem reimplementa o PortParser, e sua arquitetura e metodologia
não se baseiam nele.

*HubParser is the dependency parser developed in this master's dissertation: multi-task
BERT parsers for Brazilian Portuguese (Porttinari treebank) with linear and biaffine
heads, plus a layer-wise explainability study. PortParser is used only as a comparison
baseline; HubParser is an independent study, not an extension of it.*

## Resultados principais

| Modelo | UAS | LAS | UPOS |
|---|---|---|---|
| **BtL-Lin-MTL** (BERTimbau Large, linear, MTL) | **94,99** | **93,84** | 99,28 |
| PortParser (baseline, Lopes et al. 2024) | 96,08 | 94,61 | 99,09 |

Encoders avaliados: BERTimbau Base/Large, mBERT, JabuticaBERT.
Variantes: cabeçote {linear, biaffine} × {MTL com UPOS (cU), ablação sem UPOS (sU)}.

Do estudo de explicabilidade: a hierarquia UPOS→DEPREL→HEAD emerge em profundidade
(informação de UPOS satura na camada ~3; ligações sintáticas só se resolvem no quarto
superior da rede); um etiquetador UPOS com encoder truncado na camada 3 mantém 98,35%
de acurácia com speedup de 3,67×.

## Modelos no Hugging Face

Os 8 modelos MTL (com UPOS) foram retreinados com os melhores hiperparâmetros do Optuna,
seed 42, padding 512 e 40 épocas; o modelo publicado é o do fim da época 40. Eles estão
publicados como `guilhermedallm4/HubParser-<Encoder>-<Cabeçote>`. Teste do Porttinari
(UPOS / UAS / LAS):

| Modelo | Retreinado | Dissertação |
|---|---|---|
| [HubParser-BERTimbau-large-Linear](https://huggingface.co/guilhermedallm4/HubParser-BERTimbau-large-Linear) | 99,26 / 94,71 / 93,52 | 99,28 / 94,99 / 93,84 |
| [HubParser-BERTimbau-large-Biaffine](https://huggingface.co/guilhermedallm4/HubParser-BERTimbau-large-Biaffine) | 99,32 / 90,93 / 89,89 | 99,34 / 91,52 / 90,56 |
| [HubParser-BERTimbau-base-Linear](https://huggingface.co/guilhermedallm4/HubParser-BERTimbau-base-Linear) | 99,19 / 93,45 / 92,16 | 99,22 / 93,74 / 92,49 |
| [HubParser-BERTimbau-base-Biaffine](https://huggingface.co/guilhermedallm4/HubParser-BERTimbau-base-Biaffine) | 99,25 / 89,06 / 87,95 | 99,27 / 89,84 / 88,71 |
| [HubParser-mBERT-Linear](https://huggingface.co/guilhermedallm4/HubParser-mBERT-Linear) | 98,89 / 92,50 / 90,95 | 98,88 / 92,70 / 91,04 |
| [HubParser-mBERT-Biaffine](https://huggingface.co/guilhermedallm4/HubParser-mBERT-Biaffine) | 98,92 / 87,50 / 86,02 | 98,93 / 86,92 / 85,42 |
| [HubParser-JabuticaBERT-Linear](https://huggingface.co/guilhermedallm4/HubParser-JabuticaBERT-Linear) | 98,79 / 88,78 / 87,17 | 98,81 / 87,78 / 86,25 |
| [HubParser-JabuticaBERT-Biaffine](https://huggingface.co/guilhermedallm4/HubParser-JabuticaBERT-Biaffine) | 98,91 / 92,08 / 90,60 | 98,92 / 88,98 / 87,44 |

Uso (a entrada é uma lista de frases já tokenizadas em palavras):

```python
from transformers import AutoModel, AutoTokenizer

repo = "guilhermedallm4/HubParser-BERTimbau-large-Linear"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()

model.parse([["O", "canal", "terá", "o", "conteúdo", "reformulado", "."]], tokenizer)
# [[{'id': 1, 'form': 'O', 'upos': 'DET', 'head': 2, 'deprel': 'det'}, ...]]
```

## Experimentos multilíngues (em andamento)

A pasta [`multilingual/`](multilingual/) repete o desenho da dissertação em outras
línguas. O BERT-base-cased é treinado no inglês (UD_English-EWT) e o BETO no espanhol
(UD_Spanish-AnCora), e o mBERT treina num corpus conjunto (Porttinari + EWT + AnCora).
Todos usam os cabeçotes linear, biaffine e biaffine corrigido (alvo de head no primeiro
subtoken da palavra) e são avaliados com decodificação gulosa, Eisner e MST. O
protocolo de busca é o mesmo da dissertação. Instruções de execução: [`multilingual/INSTRUCOES_CLAUDE.md`](multilingual/INSTRUCOES_CLAUDE.md).

## Estrutura

```
HubParser/
├── notebooks/
│   ├── 01_otimizacao_treino_linear.ipynb      # Optuna + treino, cabeçote linear
│   ├── 02_otimizacao_treino_biaffine.ipynb    # Optuna + treino, cabeçote biaffine
│   ├── 03_otimizacao_ablacao.ipynb            # Optuna das ablações sem UPOS
│   ├── 04_treino_ablacao_sem_upos_linear.ipynb
│   ├── 05_treino_ablacao_sem_upos_biaffine.ipynb
│   ├── 06_inferencia_linear.ipynb             # inferência + métricas (linear)
│   ├── 07_inferencia_biaffine.ipynb           # inferência + métricas (biaffine)
│   ├── 08_inferencia_ablacao_linear.ipynb
│   ├── 09_inferencia_ablacao_biaffine.ipynb
│   ├── 10_reproducao_portparser.ipynb         # baseline PortParser
│   └── 11_explicabilidade.ipynb               # AUTOCONTIDO: logit lens, early exit,
│                                              #   block skip, probes, exit UPOS
├── scripts/                                   # análises da dissertação e do artigo
│   ├── layerwise_probes.py                    # probes por camada (5 encoders, 5 seeds)
│   ├── layerwise_frozen_heads.py              # cabeças re-treinadas no encoder congelado
│   ├── layerwise_mbert_analysis.py            # perfil por camada do mBERT
│   ├── layerwise_large_analysis.py            # perfil do BERTimbau-large (24 camadas)
│   ├── layerwise_independent_runs.py          # replicação em runs independentes
│   ├── layerwise_comparison_aaai.py           # figuras/tabelas + bootstrap pareado
│   ├── upos_early_exit.py                     # tagger UPOS truncado + benchmark
│   ├── gerar_graficos.py                      # figuras da dissertação
│   ├── label_instability_analysis.py          # instabilidade por rótulo
│   └── latex_*.py                             # tabelas LaTeX
├── artifacts/
│   ├── results/                               # JSONL de métricas por época (todos os modelos)
│   ├── layerwise/                             # CSVs da análise por camada e early exit
│   ├── figures/                               # figuras (PDF) da dissertação e do artigo
│   ├── instability/                           # resultados de instabilidade de treino
│   └── latex_tables/                          # tabelas geradas
├── requirements.txt
└── README.md
```

## Reprodução

### 1. Ambiente

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # ambiente completo
# pip install -r requirements_optuna.txt # apenas otimização/treino
```

GPU recomendada: os treinos e a análise por camada foram executados em uma
NVIDIA RTX 4090 (24 GB); os notebooks de inferência rodam em GPUs menores.

### 2. Dados e checkpoints (não versionados)

O corpus **Porttinari** deve ser obtido junto aos autores/portal do
[NILC](http://www.nilc.icmc.usp.br/) e processado para o formato HuggingFace
`datasets` esperado pelos notebooks
(`data_dois/complaints_dataset_obj_outxpos`, splits train/val/test com colunas
`tokens`, `upos`, `deprel`, `head_tags`). Os checkpoints dos modelos treinados
(`linear_BERTimbau_base/`, `biaffine_BERTImbau_base/`, etc.) são grandes demais
para o GitHub — treine-os com os notebooks 01–05 ou use as versões publicadas no
Hugging Face (seção [Modelos no Hugging Face](#modelos-no-hugging-face)).

Os notebooks e scripts localizam dados e checkpoints pela variável de ambiente
`HUBPARSER_DATA` (default: diretório pai do repositório — isto é, clone o
repositório dentro da pasta que contém `data_dois/` e os checkpoints, ou exporte
`HUBPARSER_DATA=/caminho/para/dados`).

### 3. Ordem de execução

1. **Treino**: notebooks 01–05 (Optuna: 10 trials TPE; treino final: 40 épocas).
   Logging no Weights & Biases é opcional (`WANDB_API_KEY`).
2. **Avaliação**: notebooks 06–09 reproduzem as métricas da dissertação
   (acurácia UPOS/DEPREL, UAS, LAS; 1º subtoken, decodificação gulosa, sem MST).
3. **Baseline**: o notebook 10 reproduz o PortParser no mesmo split, só para comparação.
4. **Explicabilidade**: o notebook **11** é autocontido — reproduz logit lens,
   early exit, block skip, probes por camada e o etiquetador UPOS com saída
   antecipada, gravando os CSVs em `outputs_explicabilidade/` (os valores de
   referência estão em `artifacts/layerwise/`). Os experimentos mais pesados
   (probes com 5 encoders, cabeças congeladas no modelo de 24 camadas, análises
   mBERT/large) ficam nos scripts de `scripts/`.

## Citação

Dissertação de mestrado, Universidade Federal de Pelotas (UFPel), 2026.
Artigo do estudo por camada: *"Information, Behavior, and Format: Where Syntax
Lives in BERT Depends on Who Is Asking"* (em submissão).

```bibtex
@mastersthesis{lima2026hubparser,
  title  = {HubParser: parsing de dependências multi-tarefa para o português
            brasileiro com encoders BERT},
  author = {Lima, Guilherme Dallman},
  school = {Universidade Federal de Pelotas},
  year   = {2026}
}
```
