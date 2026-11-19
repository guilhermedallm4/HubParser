// Web Worker: baixa o modelo (com cache), roda o ONNX (WebGPU ou WebAssembly) e monta as árvores.
// A interface fica livre enquanto a análise acontece.
import * as ort from "onnxruntime-web/webgpu";
import { DECODIFICACOES, LIMITES, MODELOS } from "../modelos.js";
import { conllu, montar } from "./analise.js";
import { segmentar } from "./segmentacao.js";
import { Tokenizador } from "./tokenizador.js";

// o .wasm do ONNX Runtime é empacotado pelo Vite junto com a aplicação (sem CDN)
ort.env.logLevel = "error";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CACHE = "hubparser-modelos-v1";
let ARQUIVO_ONNX = "onnx/model_fp16.onnx";
let forcarGpu = false;      // só para testes (?gpu=forcar): usa qualquer adaptador WebGPU, mesmo sem fp16
let origem = null;          // ".../{modelo}/"
let tokenizador = null;
let tokPronto = null;       // Promise do tokenizador
let gpu = { disponivel: false, f16: false, descricao: null };
let atual = null;           // { chave, sessao, cfg, dispositivo }
const configs = new Map();

const url = (modelo, arq) => origem.replace("{modelo}", modelo) + arq;
const enviar = (msg) => self.postMessage(msg);

async function abrirCache() {
  try { return await caches.open(CACHE); } catch { return null; }
}

async function emCache(modelo) {
  const c = await abrirCache();
  return !!(c && (await c.match(url(modelo, ARQUIVO_ONNX))));
}

/** Baixa um arquivo grande com progresso; guarda no Cache Storage para a próxima visita. */
async function baixar(endereco, totalEsperado, aoProgresso) {
  const cache = await abrirCache();
  const salvo = cache && (await cache.match(endereco));
  if (salvo) return new Uint8Array(await salvo.arrayBuffer());
  const r = await fetch(endereco);
  if (!r.ok) throw new Error(`Não foi possível baixar o modelo (HTTP ${r.status}).`);
  const total = Number(r.headers.get("content-length")) || totalEsperado;
  let buf = new Uint8Array(total);
  let recebido = 0;
  const leitor = r.body.getReader();
  let ultimo = 0;
  for (;;) {
    const { done, value } = await leitor.read();
    if (done) break;
    if (recebido + value.length > buf.length) {
      const maior = new Uint8Array(Math.max(buf.length * 1.25, recebido + value.length));
      maior.set(buf.subarray(0, recebido));
      buf = maior;
    }
    buf.set(value, recebido);
    recebido += value.length;
    const agora = performance.now();
    if (agora - ultimo > 120) { aoProgresso(recebido, Math.max(total, recebido)); ultimo = agora; }
  }
  aoProgresso(recebido, recebido);
  buf = buf.length === recebido ? buf : buf.slice(0, recebido);
  if (cache) {
    try { await cache.put(endereco, new Response(buf, { headers: { "content-type": "application/octet-stream" } })); }
    catch { /* sem espaço: segue sem cache */ }
  }
  return buf;
}

async function json(endereco) {
  const cache = await abrirCache();
  const salvo = cache && (await cache.match(endereco));
  if (salvo) return salvo.json();
  const r = await fetch(endereco);
  if (!r.ok) throw new Error(`Não foi possível baixar ${endereco.split("/").pop()} (HTTP ${r.status}).`);
  const texto = await r.text();
  if (cache) { try { await cache.put(endereco, new Response(texto)); } catch { /* ignora */ } }
  return JSON.parse(texto);
}

async function detectarGpu() {
  if (forcarGpu && self.navigator?.gpu && (await navigator.gpu.requestAdapter())) {
    return { disponivel: true, f16: false, descricao: "adaptador WebGPU forçado para teste" };
  }
  if (!self.navigator?.gpu) return { disponivel: false, f16: false, descricao: "Este navegador não oferece WebGPU." };
  try {
    const ad = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!ad) return { disponivel: false, f16: false, descricao: "Nenhuma placa de vídeo compatível com WebGPU." };
    const info = ad.info || {};
    const nome = [info.vendor, info.architecture || info.description].filter(Boolean).join(" ") || "GPU";
    const f16 = ad.features.has("shader-f16");
    return { disponivel: f16, f16, descricao: f16 ? nome : `${nome} sem suporte a meia precisão (shader-f16).` };
  } catch (e) {
    return { disponivel: false, f16: false, descricao: `WebGPU indisponível: ${e.message}` };
  }
}

async function iniciar(msg) {
  origem = msg.origem;
  if (msg.teste?.onnx) ARQUIVO_ONNX = `onnx/${msg.teste.onnx}`;
  forcarGpu = msg.teste?.gpu === "forcar";
  gpu = await detectarGpu();
  const baixados = [];
  for (const m of MODELOS) if (await emCache(m.id)) baixados.push(m.id);
  enviar({ id: msg.id, ok: true, resultado: { gpu, baixados } });
  esperarTokenizador().catch(() => {});
}

function esperarTokenizador() {
  // o vocabulário é o mesmo nos quatro modelos (BERTimbau); tenta de novo se a primeira carga falhar
  if (!tokPronto) {
    tokPronto = json(url("HubParser-Base-B", "tokenizer.json"))
      .then((j) => { tokenizador = new Tokenizador(j); })
      .catch((e) => { tokPronto = null; throw e; });
  }
  return tokPronto;
}

function avaliar(texto, modelo) {
  const lim = MODELOS.find((m) => m.id === modelo);
  const frases = segmentar(texto);
  const itens = [];
  const erros = [];
  let total = 0;
  frases.forEach((f, i) => {
    const n = f.palavras.length;
    const sub = tokenizador.contar(f.palavras);
    total += n;
    let motivo = null;
    if (n > lim.max_palavras) motivo = `tem ${n} palavras; o ${modelo} aceita até ${lim.max_palavras} por frase`;
    else if (sub > lim.max_subtokens) motivo = `ocupa ${sub} subtokens; o limite do encoder é ${lim.max_subtokens}`;
    if (motivo) erros.push(`A frase ${i + 1} ${motivo}.`);
    itens.push({ n_palavras: n, n_subtokens: sub, excede: motivo !== null, inicio: f.texto.slice(0, 60) });
  });
  if (frases.length > LIMITES.max_frases) erros.push(`O texto tem ${frases.length} frases; o máximo por análise é ${LIMITES.max_frases}.`);
  if (total > LIMITES.max_palavras_total) erros.push(`O texto tem ${total} palavras; o máximo por análise é ${LIMITES.max_palavras_total}.`);
  return { frases, resumo: { frases: itens, total_palavras: total, ok: erros.length === 0 && frases.length > 0, erros } };
}

async function criarSessao(bytes, dispositivo) {
  const sessao = await ort.InferenceSession.create(bytes, { graphOptimizationLevel: "all", executionProviders: [dispositivo], logSeverityLevel: 3 });
  // aquecimento: compila os shaders e revela, já aqui, GPUs que não rodam o modelo
  try {
    const ids = new ort.Tensor("int64", BigInt64Array.from([101n, 1000n, 2000n, 102n]), [1, 4]);
    const out = await sessao.run({ input_ids: ids });
    for (const v of Object.values(out)) v.dispose?.();
  } catch (e) {
    await sessao.release().catch(() => {});
    throw e;
  }
  return sessao;
}

async function carregar(id, modelo, pedido) {
  const dispositivo = pedido === "webgpu" && gpu.disponivel ? "webgpu" : "wasm";
  const chave = `${modelo}|${dispositivo}`;
  if (atual?.chave === chave) return { ...atual, carga: 0, aviso: null };
  const t0 = performance.now();
  if (atual) { await atual.sessao.release().catch(() => {}); atual = null; }
  if (!configs.has(modelo)) configs.set(modelo, await json(url(modelo, "config.json")));
  const info = MODELOS.find((m) => m.id === modelo);
  const bytes = await baixar(url(modelo, ARQUIVO_ONNX), info.mb * 2 ** 20, (recebido, total) =>
    enviar({ id, tipo: "progresso", etapa: "baixando", recebido, total }));
  enviar({ id, tipo: "progresso", etapa: "preparando", dispositivo });
  let sessao, usado = dispositivo, aviso = null;
  try {
    sessao = await criarSessao(bytes, dispositivo);
  } catch (e) {
    if (dispositivo !== "webgpu") throw e;
    console.warn("WebGPU falhou; usando a CPU.", e);
    aviso = "A placa de vídeo deste computador não conseguiu rodar o modelo. A análise seguiu no processador.";
    gpu = { ...gpu, disponivel: false, descricao: "A placa de vídeo não conseguiu rodar o modelo." };
    usado = "wasm";
    sessao = await criarSessao(bytes, "wasm");
  }
  atual = { chave: `${modelo}|${usado}`, sessao, cfg: configs.get(modelo), dispositivo: usado };
  return { ...atual, carga: performance.now() - t0, aviso };
}

async function analisar(msg) {
  await esperarTokenizador();
  const { frases, resumo } = avaliar(msg.texto, msg.modelo);
  if (!frases.length) throw new Error("Digite ao menos uma frase.");
  if (!resumo.ok) throw new Error(resumo.erros.join(" "));
  if (!(msg.decodificacao in DECODIFICACOES)) throw new Error("Decodificação desconhecida.");
  const { sessao, cfg, dispositivo, carga, aviso } = await carregar(msg.id, msg.modelo, msg.dispositivo);
  const t0 = performance.now();
  const analises = [];
  for (let i = 0; i < frases.length; i++) {
    enviar({ id: msg.id, tipo: "progresso", etapa: "analisando", frase: i + 1, frases: frases.length, dispositivo });
    const cod = tokenizador.codificar(frases[i].palavras);
    const ids = new ort.Tensor("int64", BigInt64Array.from(cod.ids, BigInt), [1, cod.ids.length]);
    const out = await sessao.run({ input_ids: ids });
    const saidas = {};
    for (const [k, v] of Object.entries(out)) {
      saidas[k] = { data: await v.getData(), dims: v.dims };
      v.dispose?.();
    }
    analises.push(montar(frases[i].palavras, cod, saidas, cfg, msg.decodificacao));
  }
  return {
    modelo: msg.modelo, decodificacao: msg.decodificacao, dispositivo, aviso, gpu,
    tempo_ms: Math.round(performance.now() - t0), carga_ms: Math.round(carga),
    frases: frases.map((f, i) => ({ texto: f.texto, palavras: analises[i], mwt: f.mwt })),
    conllu: conllu(frases, analises),
  };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.tipo === "iniciar") return await iniciar(msg);
    if (msg.tipo === "preprocessar") {
      await esperarTokenizador();
      return enviar({ id: msg.id, ok: true, resultado: avaliar(msg.texto, msg.modelo).resumo });
    }
    if (msg.tipo === "analisar") return enviar({ id: msg.id, ok: true, resultado: await analisar(msg) });
    if (msg.tipo === "baixados") {
      const baixados = [];
      for (const m of MODELOS) if (await emCache(m.id)) baixados.push(m.id);
      return enviar({ id: msg.id, ok: true, resultado: baixados });
    }
  } catch (err) {
    enviar({ id: msg.id, ok: false, erro: err?.message || String(err) });
  }
};
