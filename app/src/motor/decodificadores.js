// Decodificadores de árvore: porte linha a linha de modeling_hubparser.py (eisner, chu_liu_edmonds).
// Matrizes são arrays de Float64Array. Empates seguem o Python: np.argmax pega o primeiro máximo;
// max() de tuplas (valor, nó) desempata pelo maior nó.

const NEG = -1e9;

function argmax(arr, ini = 0, fim = arr.length) {
  let m = ini;
  for (let i = ini + 1; i < fim; i++) if (arr[i] > arr[m]) m = i;
  return m;
}

const matriz = (n, m, v) => Array.from({ length: n }, () => new Float64Array(m).fill(v));

/** (n+1)×(n+1): A[h][d] = S[d-1][h]; nada aponta para a raiz; sem laços. S: n linhas × (n+1) colunas. */
function arcMatrix(S) {
  const n = S.length;
  const A = matriz(n + 1, n + 1, NEG);
  for (let h = 0; h <= n; h++) for (let d = 1; d <= n; d++) A[h][d] = S[d - 1][h];
  for (let i = 0; i <= n; i++) A[i][i] = NEG;
  return A;
}

// ── Eisner (projetiva, exatamente um filho da raiz) ──
export function eisner(S) {
  const n = S.length;
  const A = arcMatrix(S);
  const N = n + 2;
  const idx = (s, t, d) => (s * N + t) * 2 + d;
  const C = new Float64Array(N * N * 2).fill(NEG);
  const I = new Float64Array(N * N * 2).fill(NEG);
  const bC = new Int32Array(N * N * 2);
  const bI = new Int32Array(N * N * 2);
  for (let s = 1; s <= n; s++) C[idx(s, s, 0)] = C[idx(s, s, 1)] = 0;
  for (let k = 1; k < n; k++) {
    for (let s = 1; s <= n - k; s++) {
      const t = s + k;
      // vals = C[s, s:t, 1] + C[s+1:t+1, t, 0]
      let melhor = -Infinity, r = 0;
      for (let q = 0; q < t - s; q++) {
        const v = C[idx(s, s + q, 1)] + C[idx(s + q + 1, t, 0)];
        if (v > melhor) { melhor = v; r = q; }
      }
      I[idx(s, t, 0)] = melhor + A[t][s];
      I[idx(s, t, 1)] = melhor + A[s][t];
      bI[idx(s, t, 0)] = bI[idx(s, t, 1)] = s + r;
      // vals = C[s, s:t, 0] + I[s:t, t, 0]
      melhor = -Infinity; r = 0;
      for (let q = 0; q < t - s; q++) {
        const v = C[idx(s, s + q, 0)] + I[idx(s + q, t, 0)];
        if (v > melhor) { melhor = v; r = q; }
      }
      C[idx(s, t, 0)] = melhor;
      bC[idx(s, t, 0)] = s + r;
      // vals = I[s, s+1:t+1, 1] + C[s+1:t+1, t, 1]
      melhor = -Infinity; r = 0;
      for (let q = 0; q < t - s; q++) {
        const v = I[idx(s, s + 1 + q, 1)] + C[idx(s + 1 + q, t, 1)];
        if (v > melhor) { melhor = v; r = q; }
      }
      C[idx(s, t, 1)] = melhor;
      bC[idx(s, t, 1)] = s + 1 + r;
    }
  }
  let r = 1, melhorRaiz = -Infinity;
  for (let q = 1; q <= n; q++) {
    const v = A[0][q] + C[idx(1, q, 0)] + C[idx(q, n, 1)];
    if (v > melhorRaiz) { melhorRaiz = v; r = q; }
  }
  const heads = new Array(n).fill(0);
  // backtracking iterativo (frases longas estourariam a pilha recursiva)
  const pilha = [["C", 1, r, 0], ["C", r, n, 1]];
  while (pilha.length) {
    const [tipo, s, t, d] = pilha.pop();
    if (tipo === "C") {
      if (s === t) continue;
      const m = bC[idx(s, t, d)];
      if (d === 0) pilha.push(["C", s, m, 0], ["I", m, t, 0]);
      else pilha.push(["I", s, m, 1], ["C", m, t, 1]);
    } else {
      const m = bI[idx(s, t, d)];
      if (d === 0) heads[s - 1] = t;
      else heads[t - 1] = s;
      pilha.push(["C", s, m, 1], ["C", m + 1, t, 0]);
    }
  }
  return heads;
}

// ── Chu-Liu/Edmonds (não projetiva, exatamente um filho da raiz) ──
function findCycle(heads) {
  const n = heads.length - 1;
  const cor = new Array(n + 1).fill(0);
  for (let ini = 1; ini <= n; ini++) {
    const caminho = [];
    let cur = ini;
    while (cur !== 0 && cor[cur] === 0) {
      cor[cur] = 1;
      caminho.push(cur);
      cur = heads[cur];
    }
    if (cur !== 0 && cor[cur] === 1) return caminho.slice(caminho.indexOf(cur));
    for (const p of caminho) cor[p] = 2;
  }
  return null;
}

function cle(A) {
  const n = A.length - 1;
  const heads = [0];
  for (let d = 1; d <= n; d++) {
    let m = 0;
    for (let h = 1; h <= n; h++) if (A[h][d] > A[m][d]) m = h;
    heads.push(m);
  }
  const ciclo = findCycle(heads);
  if (ciclo === null) return heads;
  const noCiclo = new Set(ciclo);
  const resto = [];
  for (let v = 0; v <= n; v++) if (!noCiclo.has(v)) resto.push(v);
  const idx = new Map(resto.map((v, i) => [v, i]));
  const c = resto.length;
  const B = matriz(c + 1, c + 1, NEG);
  const entra = new Map();
  const sai = new Map();
  let scoreCiclo = 0;
  for (const v of ciclo) scoreCiclo += A[heads[v]][v];
  for (const u of resto) {
    for (const v of resto) if (u !== v) B[idx.get(u)][idx.get(v)] = A[u][v];
    // max((ganho, v)): desempata pelo maior v
    let g = -Infinity, gv = -1;
    for (const v of ciclo) {
      const x = A[u][v] - A[heads[v]][v];
      if (x > g || (x === g && v > gv)) { g = x; gv = v; }
    }
    B[idx.get(u)][c] = scoreCiclo + g;
    entra.set(u, gv);
    if (u !== 0) {
      let s = -Infinity, w = -1;
      for (const x of ciclo) {
        if (A[x][u] > s || (A[x][u] === s && x > w)) { s = A[x][u]; w = x; }
      }
      B[c][idx.get(u)] = s;
      sai.set(u, w);
    }
  }
  const sub = cle(B);
  const out = heads.slice();
  for (const u of resto.slice(1)) {
    const h = sub[idx.get(u)];
    out[u] = h === c ? sai.get(u) : resto[h];
  }
  const u = resto[sub[c]];
  out[entra.get(u)] = u;
  return out;
}

export function chuLiuEdmonds(S) {
  const n = S.length;
  const A = arcMatrix(S);
  const heads = cle(A).slice(1);
  if (heads.filter((h) => h === 0).length === 1) return heads;
  // restrição de uma raiz: testa cada filho da raiz e fica com a melhor árvore
  let melhor = null, melhorScore = -Infinity;
  for (let r = 1; r <= n; r++) {
    const B = A.map((linha) => Float64Array.from(linha));
    B[0].fill(NEG);
    B[0][r] = A[0][r];
    const h = cle(B).slice(1);
    let sc = 0;
    for (let d = 1; d <= n; d++) sc += A[h[d - 1]][d];
    if (sc > melhorScore) { melhor = h; melhorScore = sc; }
  }
  return melhor;
}

export { argmax };
