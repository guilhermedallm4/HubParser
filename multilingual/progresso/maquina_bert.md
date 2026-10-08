# Progresso — `maquina_bert`

Atualizado em **08/10/2026 14:28** (atualização a cada hora e ao fim de cada etapa).

## Agora

- **Job:** `mbert__biaffine_fix__multilingual` — busca (validação cruzada)
- **Configuração / fold:** config 7, fold 3
- **Fold atual:** 96% (71809/74480 passos), ~02:00 restantes, 22.16 it/s
- **GPU:** NVIDIA GeForce RTX 4090, 89 %, 5628 MiB, 24564 MiB

## Jobs

| Fila | Job | Estado | Reserva | Detalhes |
|---|---|---|---|---|
| maquina_beto | `beto__linear__es` | final pronto |  | es: LAS 89.41 (gulosa) |
| maquina_beto | `beto__biaffine_fix__es` | final pronto |  | es: LAS 93.22 (gulosa) |
| maquina_beto | `beto__biaffine__es` | final pronto |  | es: LAS 87.60 (gulosa) |
| maquina_bert | `bert__linear__en` | final pronto |  | en: LAS 84.79 (gulosa) |
| maquina_bert | `bert__biaffine_fix__en` | final pronto |  | en: LAS 92.34 (gulosa) |
| maquina_bert | `bert__biaffine__en` | final pronto |  | en: LAS 80.23 (gulosa) |
| compartilhado | `mbert__linear__multilingual` | final pronto | maquina_beto | pt: LAS 93.48 en: LAS 86.98 es: LAS 88.57 (gulosa) |
| compartilhado | `mbert__biaffine_fix__multilingual` | busca 38/50 folds | maquina_bert | 58 min/fold, ~12 h restantes na busca; melhor LAS médio de validação 92.23 |
| compartilhado | `mbert__biaffine__multilingual` | busca 34/50 folds | maquina_beto | 38 min/fold, ~10 h restantes na busca; melhor LAS médio de validação 88.48 |
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
| `mbert__linear__multilingual` | pt | 95.17 / 93.48 | 95.35 / 93.62 | 95.24 / 93.53 | 98.78 |
| `mbert__linear__multilingual` | en | 89.27 / 86.98 | 89.98 / 87.59 | 89.68 / 87.32 | 96.57 |
| `mbert__linear__multilingual` | es | 91.02 / 88.57 | 91.73 / 89.22 | 91.34 / 88.86 | 98.89 |

## Eventos recentes

```
[2026-10-05 16:11:52] done final_train bert__biaffine_fix__en
[2026-10-05 16:11:53] start search bert__biaffine__en
[2026-10-06 08:54:00] start search bert__biaffine__en
[2026-10-06 16:14:28] done search bert__biaffine__en
[2026-10-06 16:14:28] start final_train bert__biaffine__en
[2026-10-06 18:03:36] done final_train bert__biaffine__en
[2026-10-06 18:03:38] claimed mbert__biaffine_fix__multilingual from the shared pool
[2026-10-06 18:03:38] start search mbert__biaffine_fix__multilingual
[2026-10-08 03:12:06] start search mbert__biaffine_fix__multilingual
[2026-10-08 13:28:41] start search mbert__biaffine_fix__multilingual
```
