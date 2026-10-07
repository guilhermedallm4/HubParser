# Progresso — `maquina_beto`

Atualizado em **07/10/2026 09:41** (atualização a cada hora e ao fim de cada etapa).

## Agora

- **Job:** `mbert__linear__multilingual` — busca (validação cruzada)
- **Configuração / fold:** config 9, fold 1
- **Fold atual:** 0% (294/74480 passos), ~30:53 restantes, 40.03 it/s
- **GPU:** NVIDIA GeForce RTX 5090, 0 %, 225 MiB, 32607 MiB

## Jobs

| Fila | Job | Estado | Reserva | Detalhes |
|---|---|---|---|---|
| maquina_beto | `beto__linear__es` | final pronto |  | es: LAS 89.41 (gulosa) |
| maquina_beto | `beto__biaffine_fix__es` | final pronto |  | es: LAS 93.22 (gulosa) |
| maquina_beto | `beto__biaffine__es` | final pronto |  | es: LAS 87.60 (gulosa) |
| maquina_bert | `bert__linear__en` | final pronto |  | en: LAS 84.79 (gulosa) |
| maquina_bert | `bert__biaffine_fix__en` | final pronto |  | en: LAS 92.34 (gulosa) |
| maquina_bert | `bert__biaffine__en` | final pronto |  | en: LAS 80.23 (gulosa) |
| compartilhado | `mbert__linear__multilingual` | busca 46/50 folds | maquina_beto | 35 min/fold, ~2 h restantes na busca; melhor LAS médio de validação 89.23 |
| compartilhado | `mbert__biaffine_fix__multilingual` | busca 15/50 folds | maquina_bert | 60 min/fold, ~35 h restantes na busca; melhor LAS médio de validação 92.16 |
| compartilhado | `mbert__biaffine__multilingual` | pendente | livre |  |
| compartilhado | `bertimbau-base__biaffine_fix__pt` | pendente | livre |  |
| compartilhado | `mbert__biaffine_fix__pt` | pendente | livre |  |
| compartilhado | `jabuticabert__biaffine_fix__pt` | pendente | livre |  |

## Resultados de teste (modelo da época 40)

| Job | Língua | Gulosa UAS / LAS | Eisner UAS / LAS | MST UAS / LAS | UPOS |
|---|---|---|---|---|---|
| `bert__biaffine__en` | en | 81.89 / 80.23 | 85.67 / 83.83 | 83.24 / 81.50 | 97.21 |
| `bert__biaffine_fix__en` | en | 94.16 / 92.34 | 94.17 / 92.31 | 94.21 / 92.37 | 97.40 |
| `bert__linear__en` | en | 86.64 / 84.79 | 88.68 / 86.66 | 87.48 / 85.55 | 97.16 |
| `beto__biaffine__es` | es | 89.54 / 87.60 | 91.28 / 89.27 | 90.12 / 88.15 | 99.08 |
| `beto__biaffine_fix__es` | es | 94.99 / 93.22 | 94.90 / 93.11 | 95.02 / 93.23 | 99.16 |
| `beto__linear__es` | es | 91.41 / 89.41 | 92.26 / 90.19 | 91.76 / 89.73 | 99.04 |

## Eventos recentes

```
[2026-10-05 06:15:24] done final_train beto__biaffine_fix__es
[2026-10-05 06:15:26] start search beto__biaffine__es
[2026-10-05 19:38:14] done search beto__biaffine__es
[2026-10-05 19:38:14] start final_train beto__biaffine__es
[2026-10-05 21:19:22] done final_train beto__biaffine__es
[2026-10-05 21:19:26] claimed mbert__linear__multilingual from the shared pool
[2026-10-05 21:19:26] start search mbert__linear__multilingual
[2026-10-06 21:28:49] FAILED search mbert__linear__multilingual (exit 1); see logs/mbert__linear__multilingual__search.log
[2026-10-07 06:10:26] start search mbert__linear__multilingual
[2026-10-07 09:41:39] start search mbert__linear__multilingual
```
