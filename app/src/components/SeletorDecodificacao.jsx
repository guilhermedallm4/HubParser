const ROTULOS = { eisner: "Eisner", mst: "MST", greedy: "Gulosa" };

export default function SeletorDecodificacao({ opcoes, valor, onChange, desabilitado }) {
  return (
    <fieldset className="grupo" disabled={desabilitado}>
      <legend>Decodificação da árvore</legend>
      <div className="segmentado" role="radiogroup">
        {Object.keys(ROTULOS).filter((k) => k in opcoes).map((k) => (
          <label key={k} className={valor === k ? "segmento ativo" : "segmento"} title={opcoes[k]}>
            <input type="radio" name="decodificacao" value={k} checked={valor === k} onChange={() => onChange(k)} />
            {ROTULOS[k]}
          </label>
        ))}
      </div>
      <p className="ajuda">{opcoes[valor]}</p>
    </fieldset>
  );
}
