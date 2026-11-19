import { forwardRef, useMemo } from "react";
import { corUpos, descreverDeprel, descreverUpos } from "../rotulos.js";

const PASSO = 34;          // altura entre níveis de arco
const LARG_CHAR = 8.4;     // largura aproximada de um caractere (15px)

// Nível de cada arco: 1 + o maior nível dos arcos estritamente dentro dele.
function niveis(palavras) {
  const arcos = palavras
    .filter((w) => w.head > 0)
    .map((w) => ({ id: w.id, a: Math.min(w.id, w.head), b: Math.max(w.id, w.head) }))
    .sort((x, y) => x.b - x.a - (y.b - y.a));
  const nivel = {};
  for (const arc of arcos) {
    let max = 0;
    for (const o of arcos) {
      if (o.id in nivel && arc.a <= o.a && o.b <= arc.b && !(o.a === arc.a && o.b === arc.b)) {
        max = Math.max(max, nivel[o.id]);
      }
    }
    nivel[arc.id] = max + 1;
  }
  return nivel;
}

const Arvore = forwardRef(function Arvore({ palavras, mwt, destaque, onDestaque }, ref) {
  const layout = useMemo(() => {
    const larguras = palavras.map(
      (w) => Math.max(w.form.length * LARG_CHAR, w.upos.length * 7, w.lemma.length * 6.6) + 28,
    );
    const x = [];
    let acc = 24;
    for (const l of larguras) {
      x.push(acc + l / 2);
      acc += l;
    }
    const nivel = niveis(palavras);
    const nivelMax = Math.max(0, ...Object.values(nivel));
    const base = PASSO * (nivelMax + 1) + 34;
    const temMwt = mwt.length > 0;
    return { x, nivel, base, largura: acc + 24, altura: base + (temMwt ? 92 : 64) };
  }, [palavras, mwt]);

  const { x, nivel, base, largura, altura } = layout;
  const ativo = (w) => destaque == null || w.id === destaque || w.head === destaque;

  return (
    <svg
      ref={ref}
      xmlns="http://www.w3.org/2000/svg"
      width={largura}
      height={altura}
      viewBox={`0 0 ${largura} ${altura}`}
      className="arvore-svg"
      role="img"
      aria-label={`Árvore de dependências: ${palavras.map((w) => w.form).join(" ")}`}
      fontFamily="'Maven Pro', system-ui, sans-serif"
    >
      <defs>
        <marker id="seta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#3b2a5c" />
        </marker>
        <marker id="seta-ativa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#7a14ff" />
        </marker>
      </defs>

      {palavras.map((w) => {
        const d = w.id - 1;
        const on = ativo(w);
        const realce = destaque != null && on;
        const cor = realce ? "#7a14ff" : "#3b2a5c";
        const opac = on ? 1 : 0.18;
        const titulo = `${w.deprel}: ${descreverDeprel(w.deprel)}`;
        if (w.head === 0) {
          return (
            <g key={`arc${w.id}`} opacity={opac}>
              <title>{`${w.form} é a raiz da frase`}</title>
              <line x1={x[d]} y1={10} x2={x[d]} y2={base - 4} stroke={cor} strokeWidth={realce ? 2.4 : 1.6}
                markerEnd={`url(#${realce ? "seta-ativa" : "seta"})`} />
              <text x={x[d] + 6} y={22} fontSize={12} fontWeight={700} fill="#c62828">root</text>
            </g>
          );
        }
        const h = w.head - 1;
        const x1 = x[h] + (h < d ? 7 : -7);
        const x2 = x[d];
        const topo = base - nivel[w.id] * PASSO;
        const mx = (x1 + x2) / 2;
        const my = base - nivel[w.id] * PASSO * 0.75;
        const lw = w.deprel.length * 7 + 10;
        return (
          <g key={`arc${w.id}`} opacity={opac}>
            <title>{titulo}</title>
            <path d={`M${x1},${base} C${x1},${topo} ${x2},${topo} ${x2},${base - 4}`} fill="none" stroke={cor}
              strokeWidth={realce ? 2.4 : 1.4} markerEnd={`url(#${realce ? "seta-ativa" : "seta"})`} />
            <rect x={mx - lw / 2} y={my - 12} width={lw} height={17} rx={8} fill={realce ? "#f1e8ff" : "#fff"} />
            <text x={mx} y={my + 1} fontSize={12} textAnchor="middle" fill={realce ? "#46009c" : "#5a3d99"}
              fontWeight={realce ? 700 : 500}>{w.deprel}</text>
          </g>
        );
      })}

      {palavras.map((w) => {
        const d = w.id - 1;
        const on = destaque == null || w.id === destaque || w.head === destaque ||
          palavras[destaque - 1]?.head === w.id;
        return (
          <g key={`w${w.id}`} className="arvore-palavra" opacity={on ? 1 : 0.35}
            onMouseEnter={() => onDestaque(w.id)} onMouseLeave={() => onDestaque(null)}
            onFocus={() => onDestaque(w.id)} onBlur={() => onDestaque(null)} tabIndex={0}>
            <title>{`${w.form} · ${descreverUpos(w.upos)} · lema "${w.lemma}"${w.feats !== "_" ? ` · ${w.feats}` : ""}`}</title>
            <rect x={x[d] - 30} y={base + 2} width={60} height={56} fill="transparent" />
            <text x={x[d]} y={base + 20} fontSize={15} textAnchor="middle" fontWeight={700}
              fill={w.id === destaque ? "#7a14ff" : "#1d1530"}>{w.form}</text>
            <text x={x[d]} y={base + 37} fontSize={11} textAnchor="middle" fontWeight={600} fill={corUpos(w.upos)}>
              {w.upos}</text>
            <text x={x[d]} y={base + 52} fontSize={11} textAnchor="middle" fill="#7a7290" fontStyle="italic">
              {w.lemma}</text>
          </g>
        );
      })}

      {mwt.map((m) => {
        const xa = x[m.inicio - 1] - 18;
        const xb = x[m.fim - 1] + 18;
        const y = base + 62;
        return (
          <g key={`mwt${m.inicio}`}>
            <title>{`"${m.forma}" é a contração de ${palavras.slice(m.inicio - 1, m.fim).map((w) => w.form).join(" + ")}`}</title>
            <path d={`M${xa},${y} L${xa},${y + 6} L${xb},${y + 6} L${xb},${y}`} fill="none" stroke="#b249f8" strokeWidth={1.2} />
            <text x={(xa + xb) / 2} y={y + 22} fontSize={12} textAnchor="middle" fill="#8a2be2" fontWeight={600}>
              {m.forma}</text>
          </g>
        );
      })}
    </svg>
  );
});

export default Arvore;
