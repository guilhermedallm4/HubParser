export function baixar(nome, conteudo, tipo = "text/plain") {
  const url = URL.createObjectURL(new Blob([conteudo], { type: `${tipo};charset=utf-8` }));
  const a = Object.assign(document.createElement("a"), { href: url, download: nome });
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Separa o CoNLL-U completo em blocos por frase.
export const blocosConllu = (conllu) => conllu.trim().split(/\n\s*\n/);
