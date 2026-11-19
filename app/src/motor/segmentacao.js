// Segmentação de texto livre no padrão do Porttinari (Universal Dependencies).
// Porte fiel de onnx/segmentacao_ref.py: \w do Python (Unicode) = [\p{L}\p{N}_].
//
// texto -> frases -> tokens -> palavras sintáticas
// * Frases: quebra em fim de linha e depois de . ! ? … quando a próxima palavra começa com
//   maiúscula, dígito, aspas ou parêntese. Abreviações comuns (Sr., Dr., Prof. ...) não encerram frase.
// * Tokens: palavras (inclusive com hífen), números (3,5 · 10.000), e-mails, URLs, reticências e
//   cada sinal de pontuação separado.
// * Palavras sintáticas: contrações de preposição + artigo/pronome são desfeitas ("da" -> "de a");
//   o token original fica registrado como multiword token (MWT) para o CoNLL-U.

const ABREVIACOES = [
  "sr", "sra", "srta", "dr", "dra", "prof", "profa", "exmo", "exma", "ilmo", "ilma", "av", "pág", "págs",
  "vol", "cap", "fig", "tab", "ed", "eng", "adv", "ltda", "obs", "aprox", "séc", "art", "inc", "nº", "núm",
  "tel", "jan", "fev", "abr", "jun", "jul", "ago", "set", "out", "nov", "dez",
];

const W = "[\\p{L}\\p{N}_]";
const NAO_W = "[^\\p{L}\\p{N}_]";
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// (?i:...) do Python: cada letra vira [xX]
const semCaixa = (s) =>
  Array.from(s).map((c) => (c.toLowerCase() !== c.toUpperCase() ? `[${c.toLowerCase()}${c.toUpperCase()}]` : esc(c))).join("");
const ABREV = [...ABREVIACOES].sort((a, b) => Array.from(b).length - Array.from(a).length).map(semCaixa).join("|");

const TOKEN_RE = new RegExp(
  [
    `(?<!${W})(?:${ABREV})\\.`,                              // abreviação com ponto (\b do Python)
    `https?://[^\\s<>"]*[^\\s<>".,;:!?)\\]}'"]`,             // URL
    `[\\p{L}\\p{N}_.+\\-]+@[\\p{L}\\p{N}_\\-]+(?:\\.[\\p{L}\\p{N}_\\-]+)+`, // e-mail
    `\\p{Nd}+(?:[.,]\\p{Nd}+)*(?!${W})`,                     // número: 3,5  10.000  1.234,56
    `${W}+(?:[\\-'’]${W}+)*`,                                // palavra, inclusive com hífen
    `\\.\\.\\.`,                                             // reticências
    `${NAO_W.slice(0, -1)}\\s]`,                             // demais sinais: [^\w\s]
  ].join("|"),
  "gu",
);

const FIM_DE_FRASE = new Set([".", "!", "?", "...", "…"]);
const FECHAMENTO = new Set(['"', "”", "’", "»", ")", "]", "}"]);
const ABRE_FRASE = '"“«(—–-';

export const CONTRACOES = {
  do: "de o", da: "de a", dos: "de os", das: "de as",
  no: "em o", na: "em a", nas: "em as",
  ao: "a o", aos: "a os", "à": "a a", "às": "a as",
  pelo: "por o", pela: "por a", pelos: "por os", pelas: "por as",
  num: "em um", numa: "em uma", nuns: "em uns", numas: "em umas",
  dum: "de um", duma: "de uma", duns: "de uns", dumas: "de umas",
  dele: "de ele", dela: "de ela", deles: "de eles", delas: "de elas",
  nele: "em ele", nela: "em ela", neles: "em eles", nelas: "em elas",
  deste: "de este", desta: "de esta", destes: "de estes", destas: "de estas", disto: "de isto",
  desse: "de esse", dessa: "de essa", desses: "de esses", dessas: "de essas", disso: "de isso",
  daquele: "de aquele", daquela: "de aquela", daqueles: "de aqueles", daquelas: "de aquelas",
  daquilo: "de aquilo",
  neste: "em este", nesta: "em esta", nestes: "em estes", nestas: "em estas", nisto: "em isto",
  nesse: "em esse", nessa: "em essa", nesses: "em esses", nessas: "em essas", nisso: "em isso",
  naquele: "em aquele", naquela: "em aquela", naqueles: "em aqueles", naquelas: "em aquelas",
  naquilo: "em aquilo",
  "àquele": "a aquele", "àquela": "a aquela", "àqueles": "a aqueles", "àquelas": "a aquelas",
  "àquilo": "a aquilo",
  daqui: "de aqui", "daí": "de aí", dali: "de ali", "dacolá": "de acolá",
};

// str.isupper() do Python: ao menos um caractere com caixa e nenhum minúsculo
const temCaixa = (c) => c.toLowerCase() !== c.toUpperCase();
export function isUpper(s) {
  let algum = false;
  for (const c of s) {
    if (!temCaixa(c)) continue;
    if (c !== c.toUpperCase()) return false;
    algum = true;
  }
  return algum;
}
const primeiro = (s) => Array.from(s)[0] ?? "";
const capitalize = (s) => { const a = Array.from(s); return a.length ? a[0].toUpperCase() + a.slice(1).join("").toLowerCase() : s; };

function frasesDaLinha(linha) {
  const toks = [...linha.matchAll(TOKEN_RE)].map((m) => [m[0], m.index, m.index + m[0].length]);
  const frases = [];
  let atual = [];
  for (let i = 0; i < toks.length; i++) {
    atual.push(toks[i]);
    if (FIM_DE_FRASE.has(toks[i][0])) {
      while (i + 1 < toks.length && FECHAMENTO.has(toks[i + 1][0])) {
        i += 1;
        atual.push(toks[i]);
      }
      const prox = i + 1 < toks.length ? primeiro(toks[i + 1][0]) : null;
      if (prox === null || isUpper(prox) || /\p{Nd}/u.test(prox) || ABRE_FRASE.includes(prox)) {
        frases.push(atual);
        atual = [];
      }
    }
  }
  if (atual.length) frases.push(atual);
  return frases;
}

function expandir(tokens) {
  const palavras = [];
  const mwt = [];
  for (const tok of tokens) {
    const exp = Object.prototype.hasOwnProperty.call(CONTRACOES, tok.toLowerCase()) ? CONTRACOES[tok.toLowerCase()] : null;
    if (exp === null) {
      palavras.push(tok);
      continue;
    }
    let partes = exp.split(" ");
    if (isUpper(tok) && Array.from(tok).length > 1) partes = partes.map((p) => p.toUpperCase()); // "DA" -> "DE A"
    else if (isUpper(primeiro(tok))) partes[0] = capitalize(partes[0]);                           // "Da" -> "De a"
    const inicio = palavras.length + 1;
    palavras.push(...partes);
    mwt.push({ inicio, fim: inicio + partes.length - 1, forma: tok });
  }
  return { palavras, mwt };
}

// str.splitlines() do Python
const QUEBRA = /\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]/;

/** Divide o texto em frases; cada frase = {texto, palavras, mwt}. */
export function segmentar(texto) {
  const saida = [];
  for (const linha of texto.split(QUEBRA)) {
    for (const toks of frasesDaLinha(linha)) {
      const { palavras, mwt } = expandir(toks.map((t) => t[0]));
      saida.push({ texto: linha.slice(toks[0][1], toks[toks.length - 1][2]), palavras, mwt });
    }
  }
  return saida;
}
