import { useState } from "react";

const BIBTEX = `@mastersthesis{lima2026hubparser,
  title   = {HubParser: parsing de dependências multi-tarefa para o
             português brasileiro com encoders BERT},
  author  = {Lima, Guilherme Dallmann},
  school  = {Programa de Pós-Graduação em Computação,
             Universidade Federal de Pelotas},
  address = {Pelotas, RS},
  year    = {2026},
  url     = {https://huggingface.co/spaces/H2IA/HubParser}
}`;

export default function Citacao() {
  const [copiado, setCopiado] = useState(false);
  const copiar = async () => {
    try {
      await navigator.clipboard.writeText(BIBTEX);
      setCopiado(true);
      setTimeout(() => setCopiado(false), 1800);
    } catch { /* sem permissão: o texto continua selecionável */ }
  };
  return (
    <section className="cartao citacao" id="como-citar" aria-labelledby="titulo-citacao">
      <h2 id="titulo-citacao">Como citar</h2>
      <p>Se o HubParser ajudou no seu trabalho, cite a dissertação em que ele foi desenvolvido:</p>
      <blockquote className="referencia">
        LIMA, Guilherme Dallmann. <strong>HubParser</strong>: parsing de dependências multi-tarefa para o português brasileiro
        com encoders BERT. 2026. Dissertação (Mestrado) – Programa de Pós-Graduação em Computação, Universidade Federal de
        Pelotas, Pelotas, 2026.
      </blockquote>
      <div className="bibtex">
        <div className="bibtex-topo">
          <span>BibTeX</span>
          <button className="botao-secundario botao-claro" onClick={copiar}>{copiado ? "Copiado ✓" : "Copiar"}</button>
        </div>
        <pre tabIndex={0}>{BIBTEX}</pre>
      </div>
      <p className="ajuda">
        Os modelos foram treinados no <strong>Porttinari</strong>, treebank do português brasileiro do projeto POeTiSA. Ao usar
        as análises em pesquisa, cite também o corpus (veja as{" "}
        <a href="https://sites.google.com/icmc.usp.br/poetisa/publications" target="_blank" rel="noreferrer">publicações do POeTiSA</a>).
      </p>
    </section>
  );
}
