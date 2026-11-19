// Modelos, decodificações e limites.

// Pasta de cada modelo: no Hub por padrão; ?modelos=/caminho/ ou VITE_MODELOS_URL trocam a origem (testes locais).
const PADRAO = "https://huggingface.co/H2IA/{modelo}/resolve/main/";
function origem() {
  try {
    const q = new URLSearchParams(globalThis.location?.search ?? "").get("modelos");
    if (q) return q.endsWith("/") ? `${q}{modelo}/` : `${q}/{modelo}/`;
  } catch { /* sem location (Node) */ }
  return import.meta.env?.VITE_MODELOS_URL ? `${import.meta.env.VITE_MODELOS_URL}{modelo}/` : PADRAO;
}
export const ORIGEM_MODELOS = origem();
export const urlModelo = (modelo, arquivo) => ORIGEM_MODELOS.replace("{modelo}", modelo) + arquivo;
export const ARQUIVO_ONNX = "onnx/model_fp16.onnx";

export const MAX_SUBTOKENS = 510; // 512 posições do BERT menos [CLS] e [SEP]

export const MODELOS = [
  { id: "HubParser-Base-B", encoder: "BERTimbau Base", cabecote: "Biaffine", las: 95.94, uas: 96.96, mb: 212,
    descricao: "Rápido e preciso. Recomendado para a maioria dos usos." },
  { id: "HubParser-Large-B", encoder: "BERTimbau Large", cabecote: "Biaffine", las: 96.16, uas: 97.2, mb: 643,
    descricao: "O mais preciso. Download maior e cerca de 3× mais lento que o Base." },
  { id: "HubParser-Base-L", encoder: "BERTimbau Base", cabecote: "Linear", las: 92.77, uas: 94.18, mb: 209,
    descricao: "Cabeçote linear: mais simples; aceita até 199 palavras por frase." },
  { id: "HubParser-Large-L", encoder: "BERTimbau Large", cabecote: "Linear", las: 93.79, uas: 95.15, mb: 639,
    descricao: "Cabeçote linear com encoder grande; até 199 palavras por frase." },
].map((m) => ({
  ...m,
  // linear: HEAD é uma classe 0..199 (0 = raiz), logo no máximo 199 palavras;
  // biaffine: o limite é o de subtokens do BERT (cada palavra tem ao menos 1 subtoken)
  max_palavras: m.cabecote === "Linear" ? 199 : MAX_SUBTOKENS,
  max_subtokens: MAX_SUBTOKENS,
}));

export const DECODIFICACOES = {
  eisner: "Eisner: melhor árvore projetiva (sem arcos cruzados). Recomendada.",
  mst: "MST (Chu-Liu-Edmonds): melhor árvore, permite arcos cruzados.",
  greedy: "Gulosa: cada palavra escolhe o pai sozinha; pode não formar uma árvore.",
};

export const LIMITES = { max_frases: 40, max_palavras_total: 1500, max_caracteres: 20000 };
