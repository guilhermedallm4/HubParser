# Instruções para executar os experimentos multilíngues do HubParser

Este arquivo é para o Claude (ou quem for operar a máquina) seguir passo a passo.
O usuário vai dizer qual é o papel da máquina: **máquina BETO** (`maquina_beto`) ou
**máquina BERT** (`maquina_bert`).

## O que mudou (versão atual)

1. **Cada encoder monolíngue treina só na própria língua.** BERT-base-cased treina e é
   avaliado só no inglês (corpus `en`), BETO só no espanhol (corpus `es`), e só o mBERT
   usa o corpus conjunto (`multilingual`).
2. **O biaffine original (`biaffine`, alvo de head desalinhado) não é mais executado.**
   Ficam apenas os cabeçotes `linear` e `biaffine_fix`.
3. **A fila foi redistribuída** (ver `jobs.json`): o BERTimbau-large com `biaffine_fix`
   no PT passou para a máquina BERT.
4. **O `run_queue.py` relê o `jobs.json` (com `git pull`) antes de cada job.** Mudanças
   futuras na fila chegam sozinhas, sem reiniciar.
5. **A busca usa uma lista fixa de 10 configurações** (`data/search_configs.json`): a
   sequência do Optuna com seed 42, igual para todos os jobs. Antes, uma reinicialização
   fazia o Optuna recomeçar a sequência e repetir configurações. A busca agora pula
   (configuração, fold) já feitos, conta repetições antigas uma vez só, completa as
   configurações que faltaram e refaz o treino final só se a melhor configuração mudar.
6. **O biaffine original voltou, mas só como comparação:** `beto__biaffine__es` e
   `bert__biaffine__en` entram na fila logo depois do `biaffine_fix` da mesma língua.
   Veja a seção "Desenho da comparação".

Se você começou a rodar com uma versão antiga:
1. Pare a fila (`kill` no PID de `logs/queue_<MAQ>.pid` e nos processos `hubparser_ml` filhos).
2. Apague só as pastas de `results/` cujo job não exista no `jobs.json` atual, como
   `results/bert__*__multilingual`. **Não** apague `bert__biaffine__en` nem
   `beto__biaffine__es`, que são jobs válidos da fila.
3. Rode `git pull`.
4. Inicie a fila de novo (passo 6). Os jobs atuais retomam do ponto onde pararam.

## Contexto

O HubParser é o parser de dependências da dissertação do usuário: um encoder BERT com
ajuste fino completo, treinado em multi-tarefa (UPOS + DEPREL + HEAD), com cabeçote
linear ou biaffine. Esta etapa testa o comportamento em outras línguas, repetindo o
desenho da dissertação (encoder monolíngue treinado na própria língua):
- **BERT-base-cased** treinado e avaliado só no inglês (UD_English-EWT, corpus `en`);
- **BETO** treinado e avaliado só no espanhol (UD_Spanish-AnCora, corpus `es`);
- **mBERT** treinado num corpus multilíngue único que junta Porttinari, EWT e AnCora
  (corpus `multilingual`) e avaliado no teste de cada língua.

Os treebanks de inglês e espanhol estão na versão UD r2.18.

São executadas duas variantes de cabeçote:
- `linear`: como na dissertação.
- `biaffine_fix`: biaffine com o alvo de head corrigido, no primeiro subtoken da
  palavra-head, e com a raiz no `[CLS]`.

Também roda o `biaffine`, o biaffine original da dissertação, cujo alvo de head é o índice
da palavra usado como posição na sequência. Por causa desse desalinhamento, ele roda só
em dois jobs, `beto__biaffine__es` (máquina BETO) e `bert__biaffine__en` (máquina BERT),
como comparação direta com o `biaffine_fix`.

A fila também inclui o `biaffine_fix` no português (só Porttinari) para BERTimbau-base,
BERTimbau-large, mBERT e JabuticaBERT, para comparar com os biaffine da dissertação.

O protocolo é o mesmo da dissertação, em `hubparser_ml/search.py` e `hubparser_ml/final_train.py`:
- **Busca:** 10 configurações fixas (`data/search_configs.json`: os 10 primeiros trials do Optuna TPE com seed 42, que são amostragem aleatória) × 5 folds sobre train+val, 40 épocas, lote 16, early stopping com paciência 5 e seleção pelo LAS médio de validação. Só `learning_rate`, `weight_decay` e `warmup_ratio` são otimizados. Usa padding dinâmico.
- **Treino final:** seed 42, lote 8, 40 épocas e padding 512. O modelo final é o da época 40.
- **Avaliação de teste:** decodificação gulosa, Eisner e MST, para cada língua.

## Regras

- **Não altere** o protocolo: hiperparâmetros, intervalos de busca, seeds, lotes, épocas e padding. Se algo parecer errado, pare e pergunte ao usuário.
- **Commits:** só o usuário aparece como autor. **Não** adicione `Co-Authored-By: Claude` nem qualquer outra atribuição ao Claude ou à Anthropic.
- O `run_queue.py` faz commit e push apenas de `multilingual/results/`. Não faça commit de dados, checkpoints ou modelos (eles estão no `.gitignore`).
- Não apague `multilingual/results/` nem `optuna.db`, porque é isso que permite retomar a execução.

## Desenho da comparação (análise prévia)

Esta rodada é uma **análise prévia do comportamento** dos cabeçotes e das decodificações.
Mais tarde, o usuário vai treinar com as partições oficiais do UD (train/dev/test), sem
validação cruzada, e comparar de novo.

- **Mesmas configurações e folds para os três cabeçotes.** Em cada língua, `linear`,
  `biaffine` e `biaffine_fix` usam as mesmas 10 configurações e os mesmos 5 folds e
  diferem só no cabeçote. Assim dá para comparar:
  - `linear` × `biaffine`: o que a dissertação comparou;
  - `biaffine` × `biaffine_fix`: o efeito isolado do bug;
  - `linear` × `biaffine_fix`: a comparação justa.
- **Um job por vez, nunca em paralelo.** Em cada máquina, os jobs rodam em sequência, na
  ordem do `jobs.json`. A GPU já fica saturada com um treino só.
- **O teste não entra na seleção.**
  - A validação cruzada usa só treino + validação: cada fold treina em 80% e é avaliado
    nos outros 20%.
  - A melhor configuração é a de maior LAS médio nos 5 folds de validação.
  - O teste é usado uma única vez por job, no treino final, depois de a configuração ter
    sido escolhida.
- **Decodificações só para análise.** O modelo final é avaliado no teste com gulosa, Eisner
  e MST apenas para medir o impacto da decodificação. Nenhuma decodificação é escolhida
  olhando o teste; essa comparação será feita depois, separadamente.
- **Não rode testes extras na GPU** enquanto a fila estiver rodando sem pedir ao usuário,
  porque eles dividem a GPU e atrasam o treino.

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
   O esperado é:
   `en {'train': 12544, 'val': 2001, 'test': 2077}`,
   `es {'train': 14287, 'val': 1654, 'test': 1721}` e
   `multilingual {'train': 32724, 'val': 4497, 'test': 5481}`.

5. **Rode o smoke test** (alguns minutos):
   ```bash
   ENC=bert; CORPUS=en     # máquina BERT  (máquina BETO: ENC=beto; CORPUS=es)
   for h in linear biaffine_fix; do
     python -m hubparser_ml.search --encoder $ENC --head $h --corpus $CORPUS --smoke &&
     python -m hubparser_ml.final_train --encoder $ENC --head $h --corpus $CORPUS --smoke || break
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
| `maquina_beto` | BETO no espanhol (linear, biaffine_fix, biaffine) → mBERT linear no corpus conjunto → biaffine_fix PT (BERTimbau-base, mBERT, JabuticaBERT) | ~3,5 dias |
| `maquina_bert` | BERT-base-cased no inglês (linear, biaffine_fix, biaffine) → mBERT biaffine_fix no corpus conjunto → biaffine_fix PT (BERTimbau-large) | ~3–3,5 dias |

Tempos medidos numa RTX 5090, por fold:
- espanhol: ~15 min;
- corpus conjunto: ~33 min.

Tempos estimados, por fold:
- inglês: ~10 min;
- Porttinari: ~4 min com encoder base e ~12 min com o large.

Cada job tem 50 folds mais um treino final de 0,7 a 4 h. Em GPUs mais lentas, os tempos
crescem na mesma proporção.
