# HubParser

Análise sintática de dependências para o português, **direto no navegador**.

O HubParser recebe um texto livre e devolve, para cada palavra, a classe gramatical (UPOS), o lema, os traços
morfológicos (FEATS) e a relação de dependência (HEAD e DEPREL), no padrão
[Universal Dependencies](https://universaldependencies.org/). Os modelos foram treinados no corpus Porttinari.
É um projeto do Hub de Inovação em Inteligência Artificial ([H2IA](https://ia.ufpel.edu.br/pt)) da
Universidade Federal de Pelotas.

**Experimente:** <https://huggingface.co/spaces/guilhermedallm4/HubParser>

A análise roda no computador de quem usa, com o [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/):
na placa de vídeo (WebGPU) quando ela suporta meia precisão, ou no processador (WebAssembly) nos demais casos.
O texto não é enviado a nenhum servidor. Na primeira análise, o navegador baixa o modelo escolhido e o guarda
para as próximas visitas.

## Modelos

| Modelo | Encoder | Cabeçote | LAS | UAS | Download (fp16) | Palavras por frase |
|---|---|---|---|---|---|---|
| [HubParser-Base-B](https://huggingface.co/guilhermedallm4/HubParser-Base-B) | BERTimbau Base | Biaffine | 95,94 | 96,96 | 212 MB | até 510 |
| [HubParser-Large-B](https://huggingface.co/guilhermedallm4/HubParser-Large-B) | BERTimbau Large | Biaffine | 96,16 | 97,20 | 643 MB | até 510 |
| [HubParser-Base-L](https://huggingface.co/guilhermedallm4/HubParser-Base-L) | BERTimbau Base | Linear | 92,77 | 94,18 | 209 MB | até 199 |
| [HubParser-Large-L](https://huggingface.co/guilhermedallm4/HubParser-Large-L) | BERTimbau Large | Linear | 93,79 | 95,15 | 639 MB | até 199 |

LAS e UAS no teste do Porttinari (1.683 frases, 33.580 palavras), com decodificação Eisner. Cada repositório de
modelo traz os pesos em PyTorch (com `model.parse(...)` em Python) e, na pasta `onnx/`, a versão fp16 usada aqui.

## Como funciona

```
texto ──► segmentação UD ──► WordPiece ──► ONNX (WebGPU ou WASM) ──► Eisner / MST / gulosa ──► árvore + CoNLL-U
          frases, tokens,     BERTimbau     logits por subtoken        no 1º subtoken de
          contrações (MWT)                                              cada palavra
```

1. **Segmentação** ([app/src/motor/segmentacao.js](app/src/motor/segmentacao.js)): divide o texto em frases e
   palavras como no Porttinari. Números (`1.234,56`), horários (`12h30`), e-mails, URLs e abreviações (`Sr.`, `Profa.`)
   ficam inteiros, e as contrações são desfeitas (`da` → `de a`) e registradas como multiword tokens.
2. **Tokenizador** ([tokenizador.js](app/src/motor/tokenizador.js)): WordPiece do BERTimbau, lido do
   `tokenizer.json` do modelo, equivalente ao `tokenizers` do Hugging Face com `is_split_into_words=True`.
3. **Modelo:** uma frase por vez. O ONNX devolve os logits de UPOS, regra de lema, FEATS e cabeça/relação.
4. **Decodificação** ([decodificadores.js](app/src/motor/decodificadores.js)): Eisner (melhor árvore projetiva),
   Chu-Liu-Edmonds (melhor árvore, com arcos cruzados) ou gulosa.
5. **Saída** ([analise.js](app/src/motor/analise.js)): lema pela regra de edição, árvore em SVG, tabela e CoNLL-U.

Tudo isso roda num Web Worker ([worker.js](app/src/motor/worker.js)), para a página não travar. O worker também
baixa o modelo, guarda-o no Cache Storage e escolhe entre WebGPU e WebAssembly. Se a GPU não conseguir rodar o
modelo, a análise segue no processador, com um aviso na tela.

## Fidelidade ao modelo original

A versão no navegador foi conferida contra o modelo em PyTorch no teste completo do Porttinari:

| Etapa | Resultado |
|---|---|
| ONNX fp32 × PyTorch | diferença 0,000 em todas as métricas e decodificações, nos 4 modelos |
| ONNX fp16 × PyTorch | diferença de no máximo 0,009 ponto percentual (Base-B: 0,000) |
| Tokenizador JS × `tokenizers` | ids idênticos nas 1.683 frases |
| Motor JS × Python (mesmo ONNX) | nenhuma palavra diferente em 33.580, nas 3 decodificações |
| Navegador × API PyTorch | CoNLL-U idêntico nos 4 modelos, em CPU e WebGPU |

A quantização int8 dinâmica também foi testada e descartada: perde de 3 a 32 pontos de LAS.

## Estrutura

```
app/      interface React + Vite e o motor do parser em JavaScript (src/motor/)
onnx/     exportação para ONNX e validação contra o PyTorch
space/    README e imagem de prévia do Space, e o script que publica a aplicação
```

## Rodar localmente

```bash
cd app
npm ci
npm run dev        # http://localhost:5173
```

Os modelos são baixados do Hugging Face. Para usar cópias locais, sirva uma pasta no formato
`<pasta>/<modelo>/{config.json,tokenizer.json,onnx/model_fp16.onnx}` e abra a página com `?modelos=/<pasta>/`.

Para publicar no Space: `npm run build` em `app/` e, na raiz, `python space/publicar.py` (com login no Hugging Face).

## Gerar e validar os ONNX

```bash
cd onnx
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

python export_onnx.py                          # os 4 modelos: saida/<modelo>/onnx/model.onnx e model_fp16.onnx
python check_onnx.py HubParser-Base-B          # compara com o PyTorch em frases aleatórias
python eval_onnx.py HubParser-Base-B model_fp16 --gold test.conllu   # métricas no teste (--ref para comparar)

python dump_tok.py --gold test.conllu          # referências do Python para o teste JS
(cd teste_js && npm install && node validar.mjs HubParser-Base-B model_fp16)
```

Os modelos em PyTorch são baixados do Hub. Com `HUBPARSER_MODELOS=<pasta>`, usam-se cópias locais. A saída vai para
`onnx/saida/` (ou `HUBPARSER_ONNX`). O teste do Porttinari vem de
[UD_Portuguese-Porttinari](https://github.com/UniversalDependencies/UD_Portuguese-Porttinari).

## Limitações

- Navegadores sem WebGPU, ou GPUs sem suporte a `shader-f16`, usam o processador, que é mais lento
  (cerca de 0,1 s por frase no Base e 0,4 s no Large, num desktop).
- Os modelos Large ocupam cerca de 640 MB no download e mais de 1 GB de memória durante a análise.
- Por análise, o limite é de 40 frases e 1.500 palavras. O cabeçote linear aceita até 199 palavras por frase, e o
  biaffine, até 510 subtokens.
- A coluna XPOS não é predita, e a decodificação gulosa pode não formar uma árvore.

## Citação

```bibtex
@mastersthesis{lima2026hubparser,
  title  = {HubParser: parsing de dependências multi-tarefa para o português
            brasileiro com encoders BERT},
  author = {Lima, Guilherme Dallman},
  school = {Universidade Federal de Pelotas},
  year   = {2026}
}
```

## Licença

[MIT](LICENSE).
