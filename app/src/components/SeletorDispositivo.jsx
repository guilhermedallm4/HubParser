// Onde o modelo roda: placa de vídeo (WebGPU) ou processador (WebAssembly).
export default function SeletorDispositivo({ gpu, valor, onChange, desabilitado }) {
  const semGpu = gpu && !gpu.disponivel;
  const opcoes = [
    ["webgpu", "GPU (WebGPU)", semGpu],
    ["wasm", "CPU", false],
  ];
  const ajuda = !gpu
    ? "Verificando o suporte a WebGPU…"
    : gpu.disponivel
      ? valor === "webgpu"
        ? `Usando a placa de vídeo: ${gpu.descricao}.`
        : "O processador funciona em qualquer navegador, mas é mais lento que a GPU."
      : `${gpu.descricao} A análise roda no processador, mais devagar.`;
  return (
    <fieldset className="grupo" disabled={desabilitado}>
      <legend>Processamento</legend>
      <div className="segmentado" role="radiogroup">
        {opcoes.map(([k, rotulo, off]) => (
          <label key={k} className={`${valor === k ? "segmento ativo" : "segmento"}${off ? " indisponivel" : ""}`}>
            <input type="radio" name="dispositivo" value={k} checked={valor === k} disabled={off} onChange={() => onChange(k)} />
            {rotulo}
          </label>
        ))}
      </div>
      <p className="ajuda">{ajuda}</p>
    </fieldset>
  );
}
