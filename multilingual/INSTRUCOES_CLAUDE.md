# Instruções para executar os experimentos multilíngues do HubParser

Este arquivo é para o Claude (ou quem for operar a máquina) seguir passo a passo.
O usuário vai dizer qual é o papel da máquina: **máquina BETO** (`maquina_beto`) ou
**máquina BERT** (`maquina_bert`).

## Contexto

O HubParser é o parser de dependências da dissertação do usuário: um encoder BERT com
ajuste fino completo, treinado em multi-tarefa (UPOS + DEPREL + HEAD), com cabeçote
linear ou biaffine. Esta etapa testa o comportamento em outras línguas. Os encoders
**BERT-base-cased** (inglês), **BETO** (espanhol) e **mBERT** são treinados num
**corpus multilíngue único**, que junta Porttinari (pt), UD_English-EWT (en) e
UD_Spanish-AnCora (es) na versão UD r2.18. A avaliação é feita no teste de cada língua.

Há três variantes de cabeçote:
- `linear`: como na dissertação.
- `biaffine`: como na dissertação. O alvo do head é o índice da palavra, usado como posição na sequência.
- `biaffine_fix`: corrigido. O alvo do head é o primeiro subtoken da palavra-head, e a raiz é o `[CLS]`.

A fila também inclui o `biaffine_fix` no português (só Porttinari) para BERTimbau-base,
BERTimbau-large, mBERT e JabuticaBERT.

O protocolo é o mesmo da dissertação, em `hubparser_ml/search.py` e `hubparser_ml/final_train.py`:
- **Busca:** Optuna TPE com 10 trials × 5 folds sobre train+val, 40 épocas, lote 16, early stopping com paciência 5 e seleção pelo LAS médio de validação. Usa padding dinâmico.
- **Treino final:** seed 42, lote 8, 40 épocas e padding 512. O modelo final é o da época 40.
- **Avaliação de teste:** decodificação gulosa, Eisner e MST, para cada língua.

## Regras

- **Não altere** o protocolo: hiperparâmetros, intervalos de busca, seeds, lotes, épocas e padding. Se algo parecer errado, pare e pergunte ao usuário.
- **Commits:** só o usuário aparece como autor. **Não** adicione `Co-Authored-By: Claude` nem qualquer outra atribuição ao Claude ou à Anthropic.
- O `run_queue.py` faz commit e push apenas de `multilingual/results/`. Não faça commit de dados, checkpoints ou modelos (eles estão no `.gitignore`).
- Não apague `multilingual/results/` nem `optuna.db`, porque é isso que permite retomar a execução.

## Passo a passo

1. **Atualize o repositório**
   ```bash
   cd <pasta do repositório HubParser>
   git pull --rebase
   cd multilingual
   ```

2. **Verifique a máquina**
   - Rode `nvidia-smi`. É preciso uma GPU com pelo menos 16 GB livres; o treino usa ~6–12 GB.
   - Rode `df -h .` e confira se há pelo menos 50 GB livres.
   - Rode `python3.10 --version`; o recomendado é Python 3.10.
   - Rode `git config user.name` e `git config user.email`; os dois precisam ser os do usuário. Se não estiverem configurados, pergunte ao usuário.
   - Confira se `git push` funciona (chave SSH ou `gh auth status`).

3. **Crie o ambiente.** Instale o PyTorch conforme a GPU:
   ```bash
   python3.10 -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   # GPUs RTX 50xx (Blackwell): CUDA 12.8
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
   # outras GPUs: pip install torch --index-url https://download.pytorch.org/whl/cu124  (ou cu121)
   pip install -r requirements.txt
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

4. **Gere os dados.** O script baixa o EWT e o AnCora r2.18 do GitHub do UD; o Porttinari já vem em `data/`:
   ```bash
   python build_data.py
   ```
   O esperado é `multilingual {'train': 32724, 'val': 4497, 'test': 5481}`.

5. **Rode o smoke test** (alguns minutos):
   ```bash
   ENC=beto   # ou bert, conforme a máquina
   for h in linear biaffine_fix; do
     python -m hubparser_ml.search --encoder $ENC --head $h --corpus multilingual --smoke &&
     python -m hubparser_ml.final_train --encoder $ENC --head $h --corpus multilingual --smoke || break
   done
   rm -rf results/*__smoke runs models
   ```
   As duas variantes precisam terminar e imprimir linhas `TEST greedy/eisner/mst`. Os
   números serão baixíssimos, o que é normal num teste com 200 frases e 2 épocas.

6. **Inicie a fila em segundo plano**, com `MAQ` igual a `maquina_beto` ou `maquina_bert`:
   ```bash
   MAQ=maquina_beto
   mkdir -p logs
   nohup .venv/bin/python run_queue.py --machine $MAQ > logs/queue_$MAQ.log 2>&1 &
   echo $! > logs/queue_$MAQ.pid
   ```

7. **Acompanhe**
   ```bash
   .venv/bin/python run_queue.py --machine $MAQ --status   # progresso, min/fold e horas restantes
   tail -n 5 logs/queue_$MAQ.log
   tail -n 3 logs/<job>__search.log
   ```
   Depois do primeiro fold, informe ao usuário quantos minutos ele levou e a estimativa total.

8. **Se cair** (queda de energia, reboot, erro): leia o log do job em `logs/` e corrija só o
   que for de ambiente (por exemplo, falta de pacote ou de disco). Depois rode de novo o
   comando do passo 6. A fila pula o que já terminou, e a busca retoma do último fold
   concluído. Em caso de erro de memória (CUDA OOM) ou de qualquer erro no código, **não**
   mude o lote nem o código: avise o usuário.

9. **Ao terminar**, o `run_queue.py` imprime o status final. Monte para o usuário uma tabela
   com UPOS, UAS e LAS por língua e por decodificação, lendo de `results/<job>/final.json`.
   Os modelos finais ficam em `models/<job>/`, fora do git; não os publique sem pedido do usuário.

## Divisão do trabalho (`jobs.json`)

| Máquina | Fila (em ordem) | Estimativa numa RTX 5090 |
|---|---|---|
| `maquina_beto` | BETO (linear, biaffine, biaffine_fix) → mBERT linear → biaffine_fix PT (BERTimbau-base, mBERT, JabuticaBERT, BERTimbau-large) | ~8 dias |
| `maquina_bert` | BERT-base-cased (linear, biaffine, biaffine_fix) → mBERT biaffine → mBERT biaffine_fix | ~8 dias |

Cada fold do corpus multilíngue leva cerca de 35–45 min numa RTX 5090, e cada job
multilíngue tem 50 folds mais um treino final de ~4 h. Em GPUs mais lentas, os tempos
crescem na mesma proporção.
