# Progresso dos experimentos multilíngues

Cada máquina publica o próprio estado de hora em hora (e ao fim de cada etapa):

- **[Máquina BETO](progresso/maquina_beto.md)**: fila `maquina_beto` e jobs do pool compartilhado que ela reservou.
- **[Máquina BERT](progresso/maquina_bert.md)**: fila `maquina_bert` e jobs do pool compartilhado que ela reservou.

Cada página mostra:
- o job e o fold que estão rodando agora, com o tempo restante e o uso da GPU;
- a tabela de todos os jobs, com o estado e quem reservou cada job do pool;
- os resultados de teste já prontos;
- os eventos recentes, como falhas, retomadas e reservas.

A tabela de jobs e os resultados são lidos do `results/` no momento da atualização, então
a página da máquina que atualizou por último é a mais recente para essas duas seções.
