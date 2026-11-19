import { UPOS } from "../rotulos.js";

export default function Legenda() {
  return (
    <details className="cartao legenda">
      <summary>Legenda das classes gramaticais (UPOS)</summary>
      <ul>
        {Object.entries(UPOS).map(([k, v]) => (
          <li key={k}><span className="pilula-upos" style={{ "--cor": v.cor }}>{k}</span> {v.nome}</li>
        ))}
      </ul>
      <p className="ajuda">Cada seta sai da palavra-pai e aponta para a palavra dependente. O rótulo no arco é a relação sintática
        (DEPREL); passe o mouse sobre ele para ver a explicação. Abaixo de cada palavra aparecem a classe e o lema. Colchetes lilases
        indicam contrações desfeitas (por exemplo, “da” = “de” + “a”).</p>
    </details>
  );
}
