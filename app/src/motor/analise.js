// Saídas do ONNX -> palavras {id, form, lemma, upos, feats, head, deprel}.
// Reproduz HubParserMixin.parse + _FullExtra.parse (modeling_hubparser.py).
import { chuLiuEdmonds, eisner } from "./decodificadores.js";

/** Regra de lema "caixa|corte|sufixo": k = forma como está, l = minúsculas. */
export function aplicarRegra(forma, regra) {
  // rule.split("|", 2) do Python: o sufixo pode conter "|"
  const i1 = regra.indexOf("|");
  const i2 = regra.indexOf("|", i1 + 1);
  const caixa = regra.slice(0, i1), corte = regra.slice(i1 + 1, i2), sufixo = regra.slice(i2 + 1);
  const f = Array.from(caixa === "k" ? forma : forma.toLowerCase());
  const c = parseInt(corte, 10);
  return (c ? f.slice(0, f.length - c) : f).join("") + sufixo;
}

/** argmax de uma linha de um tensor [1, L, K] achatado. */
function argmaxLinha(dados, K, pos) {
  const base = pos * K;
  let m = 0;
  for (let k = 1; k < K; k++) if (dados[base + k] > dados[base + m]) m = k;
  return m;
}

function logSoftmax(linha) {
  let mx = -Infinity;
  for (const v of linha) if (v > mx) mx = v;
  let soma = 0;
  const x = new Float64Array(linha.length);
  for (let i = 0; i < linha.length; i++) { x[i] = linha[i] - mx; soma += Math.exp(x[i]); }
  const ls = Math.log(soma);
  for (let i = 0; i < x.length; i++) x[i] -= ls;
  return x;
}

/**
 * palavras: formas; cod: saída de Tokenizador.codificar; saidas: {nome: {data, dims}};
 * cfg: config.json do modelo; decodificacao: greedy | eisner | mst.
 */
export function montar(palavras, cod, saidas, cfg, decodificacao) {
  const n = palavras.length;
  const fp = cod.primeiro;
  const biaffine = cfg.hubparser_head === "biaffine";
  const lab = (nome, rot) => {
    const { data, dims } = saidas[nome];
    return fp.map((p) => rot[argmaxLinha(data, dims[2], p)]);
  };
  const upos = lab("upos", cfg.upos_labels);
  const regras = lab("lemma", cfg.lemma_rules);
  const feats = lab("feats", cfg.feats_labels);

  const cabeca = saidas[biaffine ? "arc" : "head"];
  const C = cabeca.dims[2];
  const linhas = fp.map((p) => Float64Array.from(cabeca.data.subarray(p * C, (p + 1) * C)));
  const cols = biaffine ? [0, ...fp] : Array.from({ length: n + 1 }, (_, i) => i);

  let heads, colsPred;
  if (decodificacao === "greedy") {
    colsPred = linhas.map((l) => { let m = 0; for (let k = 1; k < l.length; k++) if (l[k] > l[m]) m = k; return m; });
    // biaffine: coluna -> palavra (subtoken que não é o 1º de uma palavra conta como a palavra dele)
    heads = biaffine ? colsPred.map((c) => (cod.palavraDe[c] == null ? 0 : cod.palavraDe[c] + 1)) : colsPred.slice();
  } else {
    const S = linhas.map((l) => { const ls = logSoftmax(l); return Float64Array.from(cols, (c) => ls[c]); });
    heads = decodificacao === "eisner" ? eisner(S) : chuLiuEdmonds(S);
    colsPred = heads.map((h) => (h < cols.length ? cols[h] : h));
  }

  let deprel;
  if (biaffine) {
    const { data, dims } = saidas.rel; // [1, L, L, R]
    const L = dims[1], R = dims[3];
    deprel = fp.map((p, d) => {
      const c = Math.min(colsPred[d], L - 1);
      const base = (p * L + c) * R;
      let m = 0;
      for (let k = 1; k < R; k++) if (data[base + k] > data[base + m]) m = k;
      return cfg.deprel_labels[m];
    });
  } else {
    deprel = lab("deprel", cfg.deprel_labels);
  }

  return palavras.map((form, d) => ({
    id: d + 1, form, lemma: aplicarRegra(form, regras[d]), upos: upos[d], feats: feats[d],
    head: heads[d], deprel: deprel[d],
  }));
}

/** CoNLL-U com # sent_id, # text e linhas de multiword token. */
export function conllu(frases, analises) {
  const blocos = frases.map((f, i) => {
    const linhas = [`# sent_id = ${i + 1}`, `# text = ${f.texto}`];
    const mwt = new Map(f.mwt.map((m) => [m.inicio, m]));
    for (const w of analises[i]) {
      const m = mwt.get(w.id);
      if (m) linhas.push(`${m.inicio}-${m.fim}\t${m.forma}\t_\t_\t_\t_\t_\t_\t_\t_`);
      linhas.push(`${w.id}\t${w.form}\t${w.lemma}\t${w.upos}\t_\t${w.feats}\t${w.head}\t${w.deprel}\t_\t_`);
    }
    return linhas.join("\n");
  });
  return blocos.join("\n\n") + "\n\n";
}
