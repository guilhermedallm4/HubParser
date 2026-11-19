// Barra de progresso do download do modelo e da análise.
const mb = (b) => (b / 2 ** 20).toFixed(0);

export default function Progresso({ progresso, modelo }) {
  let frac = null;
  let texto = "Preparando…";
  if (progresso?.etapa === "baixando") {
    frac = progresso.total ? progresso.recebido / progresso.total : null;
    texto = `Baixando o ${modelo.id.replace("HubParser-", "")}: ${mb(progresso.recebido)} de ${mb(progresso.total)} MB (só na primeira vez)`;
  } else if (progresso?.etapa === "preparando") {
    texto = progresso.dispositivo === "webgpu" ? "Preparando o modelo na placa de vídeo…" : "Preparando o modelo no processador…";
  } else if (progresso?.etapa === "analisando") {
    frac = (progresso.frase - 1) / progresso.frases;
    texto = `Analisando a frase ${progresso.frase} de ${progresso.frases}…`;
  }
  return (
    <div className="progresso" role="status" aria-live="polite">
      <div className={frac === null ? "progresso-barra indeterminada" : "progresso-barra"}
        role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={frac === null ? undefined : Math.round(frac * 100)}>
        <span style={frac === null ? undefined : { width: `${Math.max(2, frac * 100)}%` }} />
      </div>
      <p className="ajuda centro">{texto}</p>
    </div>
  );
}
