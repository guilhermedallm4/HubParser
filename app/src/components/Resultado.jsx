import { useRef, useState } from "react";
import Arvore from "./Arvore.jsx";
import Tabela from "./Tabela.jsx";
import { baixar } from "../util.js";

const ABAS = [
  ["arvore", "Árvore"],
  ["tabela", "Tabela"],
  ["conllu", "CoNLL-U"],
];

export default function Resultado({ frase, indice, conllu, simples = false }) {
  const [abaEscolhida, setAba] = useState("arvore");
  const aba = simples && abaEscolhida === "conllu" ? "arvore" : abaEscolhida;
  const [destaque, setDestaque] = useState(null);
  const svgRef = useRef(null);
  const id = `frase-${indice}`;

  const baixarSvg = () => {
    const xml = new XMLSerializer().serializeToString(svgRef.current);
    baixar(`hubparser-frase-${indice}.svg`, `<?xml version="1.0" encoding="UTF-8"?>\n${xml}`, "image/svg+xml");
  };

  return (
    <article className="cartao resultado" aria-labelledby={`${id}-titulo`}>
      <header className="resultado-topo">
        <div>
          <h3 id={`${id}-titulo`}>Frase {indice}</h3>
          <p className="frase-texto">“{frase.texto}”</p>
        </div>
        <div className="abas" role="tablist" aria-label={`Visualizações da frase ${indice}`}>
          {ABAS.filter(([k]) => !simples || k !== "conllu").map(([k, rotulo]) => (
            <button key={k} role="tab" id={`${id}-aba-${k}`} aria-selected={aba === k} aria-controls={`${id}-painel`}
              className={aba === k ? "aba ativa" : "aba"} onClick={() => setAba(k)}>{rotulo}</button>
          ))}
        </div>
      </header>

      <div id={`${id}-painel`} role="tabpanel" aria-labelledby={`${id}-aba-${aba}`}>
        {aba === "arvore" && (
          <>
            <div className="arvore-rolagem">
              <Arvore ref={svgRef} palavras={frase.palavras} mwt={frase.mwt} destaque={destaque} onDestaque={setDestaque} />
            </div>
            <div className="resultado-rodape">
              <span className="dica">Passe o mouse (ou use Tab) sobre uma palavra para destacar suas ligações; sobre um rótulo para ver o significado.</span>
              <button className="botao-secundario" onClick={baixarSvg}>Baixar SVG</button>
            </div>
          </>
        )}
        {aba === "tabela" && (
          <Tabela palavras={frase.palavras} mwt={frase.mwt} destaque={destaque} onDestaque={setDestaque} />
        )}
        {aba === "conllu" && <pre className="conllu" tabIndex={0}>{conllu}</pre>}
      </div>
    </article>
  );
}
