// Tokenizador WordPiece do BERTimbau (tokenizer.json), equivalente ao tokenizers do Hugging Face com
// is_split_into_words=True: BertNormalizer (clean_text, handle_chinese_chars, sem minúsculas e sem tirar
// acentos) + BertPreTokenizer + WordPiece ("##", até 100 caracteres por palavra).

const PONT_ASCII = /[!-/:-@[-`{-~]/;
const PONT = /\p{P}/u;
const OUTRO = /[\p{Cc}\p{Cf}\p{Cn}\p{Co}\p{Cs}]/u;   // char::is_other do Rust
const ESPACO_PRE = /\s/u;

function ehChines(cp) {
  return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a700 && cp <= 0x2b73f) || (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b920 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);
}

function normalizar(texto) {
  let out = "";
  for (const c of texto) {
    const cp = c.codePointAt(0);
    // mesma ordem do tokenizers: remove controles (menos \t \n \r), depois espaços viram " "
    if (cp === 0 || cp === 0xfffd || (OUTRO.test(c) && c !== "\t" && c !== "\n" && c !== "\r")) continue;
    if (ESPACO_PRE.test(c)) { out += " "; continue; }
    out += ehChines(cp) ? ` ${c} ` : c;
  }
  return out;
}

function preTokenizar(texto) {
  const partes = [];
  let atual = "";
  for (const c of texto) {
    if (ESPACO_PRE.test(c)) {
      if (atual) partes.push(atual);
      atual = "";
    } else if (PONT_ASCII.test(c) || PONT.test(c)) {
      if (atual) partes.push(atual);
      partes.push(c);
      atual = "";
    } else {
      atual += c;
    }
  }
  if (atual) partes.push(atual);
  return partes;
}

export class Tokenizador {
  constructor(tokenizerJson) {
    const m = tokenizerJson.model;
    this.vocab = new Map(Object.entries(m.vocab));
    this.unk = this.vocab.get(m.unk_token);
    this.prefixo = m.continuing_subword_prefix;
    this.maxChars = m.max_input_chars_per_word;
    this.cls = this.vocab.get("[CLS]");
    this.sep = this.vocab.get("[SEP]");
    this._cache = new Map();
  }

  wordpiece(palavra) {
    const chars = Array.from(palavra);
    if (chars.length > this.maxChars) return [this.unk];
    const ids = [];
    let ini = 0;
    while (ini < chars.length) {
      let fim = chars.length;
      let achou = null;
      while (ini < fim) {
        let sub = chars.slice(ini, fim).join("");
        if (ini > 0) sub = this.prefixo + sub;
        const id = this.vocab.get(sub);
        if (id !== undefined) { achou = id; break; }
        fim -= 1;
      }
      if (achou === null) return [this.unk];
      ids.push(achou);
      ini = fim;
    }
    return ids;
  }

  /** Subtokens de uma palavra (sem [CLS]/[SEP]). */
  palavra(p) {
    let ids = this._cache.get(p);
    if (!ids) {
      ids = preTokenizar(normalizar(p)).flatMap((x) => this.wordpiece(x));
      if (this._cache.size > 50000) this._cache.clear();
      this._cache.set(p, ids);
    }
    return ids;
  }

  /** Como tokenizer(words, is_split_into_words=True): ids com [CLS]/[SEP] e posição do 1º subtoken de cada palavra. */
  codificar(palavras) {
    const ids = [this.cls];
    const primeiro = [];
    const palavraDe = [null];
    palavras.forEach((p, i) => {
      let sub = this.palavra(p);
      if (sub.length === 0) sub = [this.unk]; // palavra sem nenhum caractere útil
      primeiro.push(ids.length);
      for (const s of sub) { ids.push(s); palavraDe.push(i); }
    });
    ids.push(this.sep);
    palavraDe.push(null);
    return { ids, primeiro, palavraDe };
  }

  contar(palavras) {
    return palavras.reduce((n, p) => n + Math.max(1, this.palavra(p).length), 0);
  }
}
