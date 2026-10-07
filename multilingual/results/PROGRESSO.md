# Progresso dos experimentos multilíngues

Atualizado em **07/10/2026 13:16** (horário de Brasília) pela máquina BERT, de hora em hora.
Os dados da máquina BETO chegam pelos envios dela (a cada 6 h e no fim de cada etapa); último envio da máquina BETO: **07/10 13:04**.

## Fila

| Job | Máquina | Estado | min/fold | Busca restante | Melhor LAS médio (validação cruzada) | LAS teste, gulosa |
|---|---|---|---|---|---|---|
| `beto__linear__es` | máquina BETO | ✅ pronto | 15 | — | 89.37 | es 89.41 |
| `beto__biaffine_fix__es` | máquina BETO | ✅ pronto | 15 | — | 93.19 | es 93.22 |
| `beto__biaffine__es` | máquina BETO | ✅ pronto | 16 | — | 86.98 | es 87.60 |
| `bert__linear__en` | máquina BERT | ✅ pronto | 18 | — | 84.05 | en 84.79 |
| `bert__biaffine_fix__en` | máquina BERT | ✅ pronto | 18 | — | 92.61 | en 92.34 |
| `bert__biaffine__en` | máquina BERT | ✅ pronto | 19 | — | 79.67 | en 80.23 |
| `mbert__linear__multilingual` | máquina BETO | 🔄 treino final | 35 | — | 89.23 | — |
| `mbert__biaffine_fix__multilingual` | máquina BERT | 🔄 busca 20/50 folds | 56 | ~28 h | 92.16 | — |
| `mbert__biaffine__multilingual` | livre | ⏳ pendente | — | — | — | — |
| `bertimbau-base__biaffine_fix__pt` | livre | ⏳ pendente | — | — | — | — |
| `mbert__biaffine_fix__pt` | livre | ⏳ pendente | — | — | — | — |
| `jabuticabert__biaffine_fix__pt` | livre | ⏳ pendente | — | — | — | — |

Cada job: 10 configurações × 5 folds (validação cruzada em train + dev) e treino final com a melhor configuração; o teste é usado uma única vez, no treino final.

## Resultados de teste (jobs prontos)

UAS / LAS por decodificação. A seleção da configuração usa só a validação cruzada; as três decodificações são só para análise.

| Encoder | Cabeçote | Corpus de treino | Teste | UPOS | Gulosa | Eisner | MST | LAS médio da validação cruzada |
|---|---|---|---|---|---|---|---|---|
| beto | linear | es | es | 99.04 | 91.41 / 89.41 | 92.26 / 90.19 | 91.76 / 89.73 | 89.37 ± 0.24 |
| beto | biaffine_fix | es | es | 99.16 | 94.99 / 93.22 | 94.90 / 93.11 | 95.02 / 93.23 | 93.19 ± 0.14 |
| beto | biaffine | es | es | 99.08 | 89.54 / 87.60 | 91.28 / 89.27 | 90.12 / 88.15 | 86.98 ± 0.35 |
| bert | linear | en | en | 97.16 | 86.64 / 84.79 | 88.68 / 86.66 | 87.48 / 85.55 | 84.05 ± 0.52 |
| bert | biaffine_fix | en | en | 97.40 | 94.16 / 92.34 | 94.17 / 92.31 | 94.21 / 92.37 | 92.61 ± 0.37 |
| bert | biaffine | en | en | 97.21 | 81.89 / 80.23 | 85.67 / 83.83 | 83.24 / 81.50 | 79.67 ± 0.54 |
