"""Referência em Python da segmentação usada no navegador (app/src/motor/segmentacao.js).

Segmentação de texto livre no padrão do Porttinari (Universal Dependencies).

texto -> frases -> tokens -> palavras sintáticas

* Frases: quebra em fim de linha e depois de . ! ? … quando a próxima palavra começa
  com maiúscula, dígito, aspas ou parêntese. Abreviações comuns (Sr., Dr., Prof. ...)
  não encerram frase.
* Tokens: palavras (inclusive com hífen), números (3,5 · 10.000), e-mails, URLs,
  reticências e cada sinal de pontuação separado.
* Palavras sintáticas: contrações de preposição + artigo/pronome são desfeitas, como
  no Porttinari ("da" -> "de a"); o token original fica registrado como multiword
  token (MWT) para o CoNLL-U.

"nos" não é expandido (pode ser pronome: "ele nos viu") e clíticos com hífen
("fazê-lo") ficam inteiros.
"""
import re
from dataclasses import dataclass, field

ABREVIACOES = (
    "sr", "sra", "srta", "dr", "dra", "prof", "profa", "exmo", "exma", "ilmo", "ilma", "av", "pág", "págs",
    "vol", "cap", "fig", "tab", "ed", "eng", "adv", "ltda", "obs", "aprox", "séc", "art", "inc", "nº", "núm",
    "tel", "jan", "fev", "abr", "jun", "jul", "ago", "set", "out", "nov", "dez",
)
_ABREV = "|".join(sorted(map(re.escape, ABREVIACOES), key=len, reverse=True))

TOKEN_RE = re.compile(
    rf"""
      (?i:\b(?:{_ABREV})\.)                                  # abreviação com ponto
    | https?://[^\s<>"]*[^\s<>".,;:!?)\]}}'"]                # URL
    | [\w.+-]+@[\w-]+(?:\.[\w-]+)+                           # e-mail
    | \d+(?:[.,]\d+)*(?!\w)                                  # número: 3,5  10.000  1.234,56
    | \w+(?:[-'’]\w+)*                                       # palavra, inclusive com hífen
    | \.\.\.                                                 # reticências
    | [^\w\s]                                                # demais sinais
    """,
    re.X,
)

FIM_DE_FRASE = {".", "!", "?", "...", "…"}
FECHAMENTO = {'"', "”", "’", "»", ")", "]", "}"}

CONTRACOES = {
    "do": "de o", "da": "de a", "dos": "de os", "das": "de as",
    "no": "em o", "na": "em a", "nas": "em as",
    "ao": "a o", "aos": "a os", "à": "a a", "às": "a as",
    "pelo": "por o", "pela": "por a", "pelos": "por os", "pelas": "por as",
    "num": "em um", "numa": "em uma", "nuns": "em uns", "numas": "em umas",
    "dum": "de um", "duma": "de uma", "duns": "de uns", "dumas": "de umas",
    "dele": "de ele", "dela": "de ela", "deles": "de eles", "delas": "de elas",
    "nele": "em ele", "nela": "em ela", "neles": "em eles", "nelas": "em elas",
    "deste": "de este", "desta": "de esta", "destes": "de estes", "destas": "de estas", "disto": "de isto",
    "desse": "de esse", "dessa": "de essa", "desses": "de esses", "dessas": "de essas", "disso": "de isso",
    "daquele": "de aquele", "daquela": "de aquela", "daqueles": "de aqueles", "daquelas": "de aquelas",
    "daquilo": "de aquilo",
    "neste": "em este", "nesta": "em esta", "nestes": "em estes", "nestas": "em estas", "nisto": "em isto",
    "nesse": "em esse", "nessa": "em essa", "nesses": "em esses", "nessas": "em essas", "nisso": "em isso",
    "naquele": "em aquele", "naquela": "em aquela", "naqueles": "em aqueles", "naquelas": "em aquelas",
    "naquilo": "em aquilo",
    "àquele": "a aquele", "àquela": "a aquela", "àqueles": "a aqueles", "àquelas": "a aquelas",
    "àquilo": "a aquilo",
    "daqui": "de aqui", "daí": "de aí", "dali": "de ali", "dacolá": "de acolá",
}


@dataclass
class Frase:
    texto: str                                   # trecho original
    palavras: list = field(default_factory=list)  # palavras sintáticas (entrada do parser)
    mwt: list = field(default_factory=list)       # (id_inicio, id_fim, forma original)


def _frases_da_linha(linha):
    """Lista de frases; cada frase é uma lista de (token, início, fim) na linha."""
    toks = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(linha)]
    atual, frases, i = [], [], 0
    while i < len(toks):
        atual.append(toks[i])
        if toks[i][0] in FIM_DE_FRASE:
            while i + 1 < len(toks) and toks[i + 1][0] in FECHAMENTO:      # fim." ou fim.)
                i += 1
                atual.append(toks[i])
            prox = toks[i + 1][0] if i + 1 < len(toks) else None
            if prox is None or prox[0].isupper() or prox[0].isdigit() or prox[0] in '"“«(—–-':
                frases.append(atual)
                atual = []
        i += 1
    if atual:
        frases.append(atual)
    return frases


def _expandir(tokens):
    palavras, mwt = [], []
    for tok in tokens:
        exp = CONTRACOES.get(tok.lower())
        if exp is None:
            palavras.append(tok)
            continue
        partes = exp.split()
        if tok.isupper() and len(tok) > 1:              # "DA" -> "DE A"
            partes = [p.upper() for p in partes]
        elif tok[0].isupper():                          # "Da" -> "De a"
            partes[0] = partes[0].capitalize()
        inicio = len(palavras) + 1
        palavras.extend(partes)
        mwt.append((inicio, inicio + len(partes) - 1, tok))
    return palavras, mwt


def segmentar(texto: str) -> list[Frase]:
    """Divide o texto em frases e cada frase em palavras sintáticas."""
    saida = []
    for linha in texto.splitlines():
        for toks in _frases_da_linha(linha):
            palavras, mwt = _expandir([t for t, _, _ in toks])
            saida.append(Frase(texto=linha[toks[0][1]:toks[-1][2]], palavras=palavras, mwt=mwt))
    return saida
