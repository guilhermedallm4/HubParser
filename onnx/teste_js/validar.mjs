// Confere o motor JS (frontend/src/motor) contra o Python, no Node com onnxruntime-web (WebAssembly).
// Antes: python dump_tok.py --gold test.conllu  e  python eval_onnx.py <modelo> <variante> --gold test.conllu
// uso: node validar.mjs HubParser-Base-B model_fp16 [limite]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-web";
import { montar } from "../../app/src/motor/analise.js";
import { segmentar } from "../../app/src/motor/segmentacao.js";
import { Tokenizador } from "../../app/src/motor/tokenizador.js";

const [nome, variante, lim] = process.argv.slice(2);
const AQUI = path.dirname(fileURLToPath(import.meta.url));
const ONNX = process.env.HUBPARSER_ONNX || path.join(AQUI, "..", "saida");
// config.json e tokenizer.json: pasta local (HUBPARSER_MODELOS) ou o Hub
async function lerJson(arquivo) {
  const local = process.env.HUBPARSER_MODELOS && path.join(process.env.HUBPARSER_MODELOS, nome, arquivo);
  if (local && fs.existsSync(local)) return JSON.parse(fs.readFileSync(local));
  const r = await fetch(`https://huggingface.co/H2IA/${nome}/resolve/main/${arquivo}`);
  return r.json();
}
const cfg = await lerJson("config.json");
const tok = new Tokenizador(await lerJson("tokenizer.json"));
const ref = JSON.parse(fs.readFileSync(`${ONNX}/ref_tok.json`));

// 1. tokenizador
let difTok = 0;
ref.sents.forEach((s, i) => {
  const a = tok.codificar(s).ids.join(" ");
  if (a !== ref.ids[i].join(" ")) {
    difTok++;
    if (difTok <= 5) console.log("TOKEN difere:", JSON.stringify(s).slice(0, 200), "\n js:", a.slice(0, 200), "\n py:", ref.ids[i].join(" ").slice(0, 200));
  }
});
console.log(`tokenizador: ${ref.sents.length - difTok}/${ref.sents.length} frases idênticas`);

// 2. segmentação
let difSeg = 0;
ref.textos.forEach((t, i) => {
  const a = JSON.stringify(segmentar(t));
  const b = JSON.stringify(ref.segs[i]);
  if (a !== b) { difSeg++; console.log("SEGMENTAÇÃO difere:\n js:", a, "\n py:", b); }
});
console.log(`segmentação: ${ref.textos.length - difSeg}/${ref.textos.length} textos idênticos`);

if (variante) {
  // 3. modelo + decodificadores + pós-processamento
  ort.env.wasm.numThreads = 8;
  const sess = await ort.InferenceSession.create(`${ONNX}/${nome}/onnx/${variante}.onnx`, { executionProviders: ["wasm"] });
  const pred = JSON.parse(fs.readFileSync(`${ONNX}/${nome}/pred_${variante}.json`));
  const n = lim ? Number(lim) : pred.length;
  const campos = ["upos", "lemma", "feats"];
  const dif = { upos: 0, lemma: 0, feats: 0, greedy: 0, eisner: 0, mst: 0 };
  let palavras = 0;
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    const s = ref.sents[i];
    const cod = tok.codificar(s);
    const ids = new ort.Tensor("int64", BigInt64Array.from(cod.ids, BigInt), [1, cod.ids.length]);
    const out = await sess.run({ input_ids: ids });
    const saidas = Object.fromEntries(Object.entries(out).map(([k, v]) => [k, { data: v.data, dims: v.dims }]));
    palavras += s.length;
    for (const dec of ["greedy", "eisner", "mst"]) {
      const ws = montar(s, cod, saidas, cfg, dec);
      ws.forEach((w, j) => {
        if (dec === "eisner") for (const c of campos) if (w[c] !== pred[i][c][j]) dif[c]++;
        if (w.head !== pred[i][dec].head[j] || w.deprel !== pred[i][dec].deprel[j]) dif[dec]++;
      });
    }
  }
  const seg = (performance.now() - t0) / 1000;
  console.log(`${nome} ${variante}: ${n} frases, ${palavras} palavras, ${seg.toFixed(1)} s (${(seg / n * 1000).toFixed(0)} ms/frase)`);
  console.log("palavras diferentes do Python:", JSON.stringify(dif));
}
