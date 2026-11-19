export default function SeletorModelo({ modelos, valor, onChange, desabilitado, baixados = [] }) {
  return (
    <fieldset className="grupo" disabled={desabilitado}>
      <legend>Modelo</legend>
      <div className="opcoes-modelo">
        {modelos.map((m) => (
          <label key={m.id} className={valor === m.id ? "opcao-modelo ativa" : "opcao-modelo"}>
            <input type="radio" name="modelo" value={m.id} checked={valor === m.id} onChange={() => onChange(m.id)} />
            <span className="opcao-titulo">
              {m.id.replace("HubParser-", "")}
              {m.id === "HubParser-Base-B" && <span className="selo">recomendado</span>}
              {baixados.includes(m.id) && <span className="selo selo-baixado" title="Já está guardado neste navegador">baixado</span>}
            </span>
            <span className="opcao-sub">{m.encoder} · {m.cabecote} · {m.mb} MB</span>
            <span className="opcao-metricas">LAS {m.las.toFixed(2).replace(".", ",")} · até {m.max_palavras} palavras/frase</span>
            <span className="opcao-desc">{m.descricao}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
