export default function Contador({ previa, modelo, carregando, compacto = false }) {
  if (!previa || previa.frases.length === 0) {
    return <p className="contador-vazio">O texto é dividido em frases, e cada frase em palavras e sinais de pontuação, como na Universal Dependencies.</p>;
  }
  const limite = modelo.max_palavras;
  if (compacto) {
    return (
      <div className="contador" aria-live="polite" aria-busy={carregando}>
        <p className="contador-resumo">
          <strong>{previa.frases.length}</strong> {previa.frases.length === 1 ? "frase" : "frases"} ·{" "}
          <strong>{previa.total_palavras}</strong> palavras e sinais de pontuação
        </p>
        {previa.erros.length > 0 && (
          <ul className="erros" role="alert">{previa.erros.map((e) => <li key={e}>{e}</li>)}</ul>
        )}
      </div>
    );
  }
  return (
    <div className="contador" aria-live="polite" aria-busy={carregando}>
      <p className="contador-resumo">
        <strong>{previa.frases.length}</strong> {previa.frases.length === 1 ? "frase" : "frases"} ·{" "}
        <strong>{previa.total_palavras}</strong> palavras e sinais · limite de <strong>{limite}</strong> por frase no {modelo.id.replace("HubParser-", "")}
      </p>
      <ol className="barras">
        {previa.frases.map((f, i) => {
          const uso = Math.max(f.n_palavras / limite, f.n_subtokens / modelo.max_subtokens);
          const estado = f.excede ? "excede" : uso >= 0.8 ? "perto" : "ok";
          return (
            <li key={i} className={`barra-item ${estado}`}>
              <span className="barra-rotulo" title={f.inicio}>Frase {i + 1}</span>
              <span className="barra" aria-hidden="true"><span style={{ width: `${Math.min(100, uso * 100)}%` }} /></span>
              <span className="barra-valor">{f.n_palavras}/{limite}</span>
            </li>
          );
        })}
      </ol>
      {previa.erros.length > 0 && (
        <ul className="erros" role="alert">
          {previa.erros.map((e) => <li key={e}>{e}</li>)}
        </ul>
      )}
    </div>
  );
}
