// Ponte entre a interface e o Web Worker do parser.
import { ORIGEM_MODELOS } from "../modelos.js";

let worker = null;
let seq = 0;
const pendentes = new Map();

function obterWorker() {
  if (!worker) {
    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const msg = e.data;
      const p = pendentes.get(msg.id);
      if (!p) return;
      if (msg.tipo === "progresso") return p.aoProgresso?.(msg);
      pendentes.delete(msg.id);
      if (msg.ok) p.resolve(msg.resultado);
      else p.reject(new Error(msg.erro));
    };
    worker.onerror = (e) => {
      for (const p of pendentes.values()) p.reject(new Error(e.message || "Falha no processamento local."));
      pendentes.clear();
    };
  }
  return worker;
}

function pedir(tipo, dados = {}, aoProgresso) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pendentes.set(id, { resolve, reject, aoProgresso });
    obterWorker().postMessage({ id, tipo, ...dados });
  });
}

// parâmetros só para testes: ?onnx=model.onnx (outro arquivo ONNX) e ?gpu=forcar
function parametrosTeste() {
  const q = new URLSearchParams(location.search);
  return { onnx: q.get("onnx"), gpu: q.get("gpu") };
}

export const iniciar = () => pedir("iniciar", { origem: ORIGEM_MODELOS, teste: parametrosTeste() });
export const preprocessar = (texto, modelo) => pedir("preprocessar", { texto, modelo });
export const analisar = (texto, modelo, decodificacao, dispositivo, aoProgresso) =>
  pedir("analisar", { texto, modelo, decodificacao, dispositivo }, aoProgresso);
export const modelosBaixados = () => pedir("baixados");
