import { useState } from "react";
import { DEPREL, FEATS, GRUPOS_DEPREL, UPOS } from "../rotulos.js";

// As 16 classes que o modelo prediz (o Porttinari não usa PART).
const CLASSES = Object.keys(UPOS).filter((k) => k !== "PART");

const ABAS = [
  ["upos", "Classes gramaticais", "UPOS"],
  ["deprel", "Relações sintáticas", "DEPREL"],
  ["feats", "Traços morfológicos", "FEATS"],
];

export default function Legenda() {
  const [aba, setAba] = useState("upos");
  return (
    <details className="cartao legenda" id="guia" open={typeof location !== "undefined" && location.hash === "#guia"}>
      <summary>
        <span className="legenda-titulo">Guia de leitura</span>
        <span className="legenda-sub">classes gramaticais, relações sintáticas e traços morfológicos</span>
      </summary>

      <p className="legenda-como">
        Na árvore, cada seta sai da palavra que manda (o <em>núcleo</em>) e aponta para a palavra que depende dela. O rótulo da
        seta é a relação sintática. Embaixo de cada palavra aparecem a classe gramatical e o lema. Colchetes lilases marcam
        contrações desfeitas, como “da” = “de” + “a”. Na aba <strong>Tabela</strong> aparecem também os traços morfológicos.
      </p>

      <div className="abas legenda-abas" role="tablist" aria-label="Partes do guia">
        {ABAS.map(([k, rotulo, sigla]) => (
          <button key={k} role="tab" aria-selected={aba === k} aria-controls="guia-painel" id={`guia-aba-${k}`}
            className={aba === k ? "aba ativa" : "aba"} onClick={() => setAba(k)}>
            {rotulo} <span className="sigla">{sigla}</span>
          </button>
        ))}
      </div>

      <div id="guia-painel" role="tabpanel" aria-labelledby={`guia-aba-${aba}`}>
        {aba === "upos" && (
          <ul className="legenda-upos">
            {CLASSES.map((k) => (
              <li key={k}><span className="pilula-upos" style={{ "--cor": UPOS[k].cor }}>{k}</span> {UPOS[k].nome}</li>
            ))}
          </ul>
        )}

        {aba === "deprel" && (
          <div className="legenda-grupos">
            {GRUPOS_DEPREL.map(([grupo, rels]) => (
              <section key={grupo}>
                <h4>{grupo}</h4>
                <dl>
                  {rels.map((r) => (
                    <div key={r}><dt><code>{r}</code></dt><dd>{DEPREL[r]}</dd></div>
                  ))}
                </dl>
              </section>
            ))}
          </div>
        )}

        {aba === "feats" && (
          <>
            <p className="ajuda">Cada palavra recebe um conjunto de traços no formato <code>Traço=Valor</code>, por exemplo
              <code> Gender=Fem|Number=Sing</code> para “casa”. Palavras sem flexão ficam com “_”.</p>
            <dl className="legenda-feats">
              {Object.entries(FEATS).map(([k, [nome, valores]]) => (
                <div key={k}>
                  <dt><code>{k}</code> {nome}</dt>
                  <dd>{Object.entries(valores).map(([v, d]) => (
                    <span key={v} className="traco"><code>{v}</code> {d}</span>
                  ))}</dd>
                </div>
              ))}
            </dl>
          </>
        )}
      </div>
    </details>
  );
}
