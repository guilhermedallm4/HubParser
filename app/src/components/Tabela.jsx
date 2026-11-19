import { corUpos, descreverDeprel, descreverUpos } from "../rotulos.js";

export default function Tabela({ palavras, mwt, destaque, onDestaque }) {
  const contracao = Object.fromEntries(mwt.flatMap((m) => {
    const ids = [];
    for (let i = m.inicio; i <= m.fim; i++) ids.push([i, m.forma]);
    return ids;
  }));
  return (
    <div className="tabela-rolagem">
      <table className="tabela">
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Palavra</th>
            <th scope="col">Lema</th>
            <th scope="col">Classe (UPOS)</th>
            <th scope="col">Traços (FEATS)</th>
            <th scope="col">Pai</th>
            <th scope="col">Relação (DEPREL)</th>
          </tr>
        </thead>
        <tbody>
          {palavras.map((w) => {
            const pai = w.head === 0 ? null : palavras[w.head - 1];
            const ligada = destaque != null && (w.id === destaque || w.head === destaque);
            return (
              <tr key={w.id} className={w.id === destaque ? "linha-ativa" : ligada ? "linha-ligada" : ""}
                onMouseEnter={() => onDestaque(w.id)} onMouseLeave={() => onDestaque(null)}>
                <td className="num">{w.id}</td>
                <td>
                  <strong>{w.form}</strong>
                  {contracao[w.id] && <span className="etiqueta-mwt" title={`parte da contração "${contracao[w.id]}"`}>{contracao[w.id]}</span>}
                </td>
                <td><em>{w.lemma}</em></td>
                <td>
                  <span className="pilula-upos" style={{ "--cor": corUpos(w.upos) }} title={descreverUpos(w.upos)}>{w.upos}</span>
                  <span className="desc">{descreverUpos(w.upos)}</span>
                </td>
                <td className="feats">
                  {w.feats === "_" ? <span className="vazio">—</span> :
                    w.feats.split("|").map((f) => <span key={f} className="traco">{f.replace("=", ": ")}</span>)}
                </td>
                <td>{pai ? <>{w.head} <span className="desc">({pai.form})</span></> : <span className="raiz">0 · raiz</span>}</td>
                <td><code>{w.deprel}</code> <span className="desc">{descreverDeprel(w.deprel)}</span></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
