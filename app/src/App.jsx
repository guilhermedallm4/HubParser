import { useEffect, useMemo, useRef, useState } from "react";
import Contador from "./components/Contador.jsx";
import Legenda from "./components/Legenda.jsx";
import Resultado from "./components/Resultado.jsx";
import Progresso from "./components/Progresso.jsx";
import SeletorDecodificacao from "./components/SeletorDecodificacao.jsx";
import SeletorDispositivo from "./components/SeletorDispositivo.jsx";
import SeletorModelo from "./components/SeletorModelo.jsx";
import * as motor from "./motor/cliente.js";
import { DECODIFICACOES, LIMITES, MODELOS } from "./modelos.js";
import { EXEMPLOS } from "./rotulos.js";
import { baixar, blocosConllu } from "./util.js";

export default function App() {
  const [ambiente, setAmbiente] = useState(null); // { gpu, baixados }
  const [erroConfig, setErroConfig] = useState(null);
  const [dispositivo, setDispositivo] = useState("webgpu");
  const [progresso, setProgresso] = useState(null);
  const [texto, setTexto] = useState(EXEMPLOS[0]);
  const [modelo, setModelo] = useState("HubParser-Base-B");
  const [decod, setDecod] = useState("eisner");
  const [previa, setPrevia] = useState(null);
  const [contando, setContando] = useState(false);
  const [analisando, setAnalisando] = useState(false);
  const [resultado, setResultado] = useState(null);
  const [erro, setErro] = useState(null);
  const [copiado, setCopiado] = useState(false);
  const resultadosRef = useRef(null);

  useEffect(() => {
    motor.iniciar()
      .then((amb) => {
        setAmbiente(amb);
        setDispositivo(amb.gpu.disponivel ? "webgpu" : "wasm");
      })
      .catch((e) => setErroConfig(`Não foi possível iniciar o HubParser neste navegador: ${e.message}`));
  }, []);

  // contador ao vivo: segmenta o texto (mesma regra da análise), com atraso de 300 ms
  useEffect(() => {
    if (!texto.trim()) {
      setPrevia(null);
      return;
    }
    let valido = true;
    setContando(true);
    const t = setTimeout(() => {
      motor.preprocessar(texto, modelo)
        .then((r) => valido && setPrevia(r))
        .catch(() => {})
        .finally(() => valido && setContando(false));
    }, 300);
    return () => {
      valido = false;
      clearTimeout(t);
    };
  }, [texto, modelo]);

  const infoModelo = useMemo(() => MODELOS.find((m) => m.id === modelo), [modelo]);
  const podeAnalisar = !!texto.trim() && !analisando && (previa ? previa.ok : true);

  async function analisar() {
    if (!podeAnalisar) return;
    setAnalisando(true);
    setErro(null);
    setProgresso(null);
    try {
      const r = await motor.analisar(texto, modelo, decod, dispositivo, setProgresso);
      setResultado(r);
      setAmbiente((a) => a && { gpu: r.gpu, baixados: a.baixados.includes(modelo) ? a.baixados : [...a.baixados, modelo] });
      if (!r.gpu.disponivel) setDispositivo("wasm");
      requestAnimationFrame(() => resultadosRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } catch (e) {
      setErro(e.message);
    } finally {
      setAnalisando(false);
      setProgresso(null);
    }
  }

  const aoTeclar = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      analisar();
    }
  };

  async function copiarConllu() {
    try {
      await navigator.clipboard.writeText(resultado.conllu);
      setCopiado(true);
      setTimeout(() => setCopiado(false), 1800);
    } catch {
      setErro("Não foi possível copiar. Use o botão de download.");
    }
  }

  const blocos = resultado ? blocosConllu(resultado.conllu) : [];

  return (
    <>
      <a className="pular" href="#conteudo">Pular para o conteúdo</a>
      <header className="topo">
        <div className="topo-conteudo">
          <a href="https://ia.ufpel.edu.br/pt" className="logo" aria-label="H2IA: Hub de Inovação em Inteligência Artificial da UFPel">
            <img src="/h2ia-logo.png" alt="H2IA" width="120" height="37" />
          </a>
          <div className="topo-titulo">
            <h1>HubParser</h1>
            <p>Análise sintática de dependências para o português</p>
          </div>
        </div>
      </header>

      <main id="conteudo" className="conteudo">
        <section className="intro">
          <h2>Da frase à árvore sintática</h2>
          <p>
            Digite um texto em português. O HubParser identifica, para cada palavra, a <strong>classe gramatical</strong>,
            o <strong>lema</strong>, os <strong>traços morfológicos</strong> e a <strong>relação sintática</strong> com a
            palavra de que depende, no padrão <a href="https://universaldependencies.org/" target="_blank" rel="noreferrer">Universal Dependencies</a>.
          </p>
          <p className="privacidade">
            <span aria-hidden="true">🔒</span> A análise roda <strong>no seu navegador</strong>, na placa de vídeo (WebGPU) ou no
            processador. O texto não é enviado a nenhum servidor; só o modelo é baixado na primeira vez e fica guardado.
          </p>
        </section>

        {erroConfig && <div className="aviso erro" role="alert">{erroConfig}</div>}

        <section className="cartao entrada" aria-labelledby="titulo-entrada">
          <div className="entrada-texto">
            <label id="titulo-entrada" htmlFor="texto" className="rotulo">Texto</label>
            <textarea
              id="texto"
              value={texto}
              onChange={(e) => setTexto(e.target.value)}
              onKeyDown={aoTeclar}
              rows={6}
              maxLength={LIMITES.max_caracteres}
              placeholder="Ex.: O restaurante universitário da UFPel serve almoço e jantar."
              aria-describedby="contador ajuda-texto"
              spellCheck
            />
            <div className="exemplos" aria-label="Exemplos">
              <span>Exemplos:</span>
              {EXEMPLOS.map((ex, i) => (
                <button key={i} type="button" className="chip" onClick={() => setTexto(ex)} title={ex}>
                  {ex.length > 38 ? `${ex.slice(0, 36)}…` : ex}
                </button>
              ))}
              {texto && <button type="button" className="chip chip-limpar" onClick={() => setTexto("")}>Limpar</button>}
            </div>
            <div id="contador">{infoModelo && <Contador previa={previa} modelo={infoModelo} carregando={contando} />}</div>
          </div>

          <div className="entrada-opcoes">
            <SeletorModelo modelos={MODELOS} valor={modelo} onChange={setModelo} desabilitado={analisando}
              baixados={ambiente?.baixados ?? []} />
            <SeletorDecodificacao opcoes={DECODIFICACOES} valor={decod} onChange={setDecod} desabilitado={analisando} />
            <SeletorDispositivo gpu={ambiente?.gpu} valor={dispositivo} onChange={setDispositivo} desabilitado={analisando || !ambiente} />
            <button className="botao-principal" onClick={analisar} disabled={!podeAnalisar} aria-busy={analisando}>
              {analisando ? <><span className="spinner" aria-hidden="true" /> {progresso?.etapa === "baixando" ? "Baixando o modelo…" : progresso?.etapa === "preparando" ? "Preparando o modelo…" : "Analisando…"}</> : "Analisar"}
            </button>
            {analisando && <Progresso progresso={progresso} modelo={infoModelo} />}
            {!analisando && infoModelo && !ambiente?.baixados.includes(modelo) && (
              <p className="ajuda centro">Na primeira análise, o {modelo.replace("HubParser-", "")} baixa cerca de {infoModelo.mb} MB.</p>
            )}
            <p id="ajuda-texto" className="ajuda centro">Atalho: <kbd>Ctrl</kbd> + <kbd>Enter</kbd></p>
          </div>
        </section>

        {erro && <div className="aviso erro" role="alert">{erro}</div>}
        {resultado?.aviso && <div className="aviso alerta" role="status">{resultado.aviso}</div>}

        <section ref={resultadosRef} className="resultados" aria-live="polite" aria-label="Resultados">
          {resultado && (
            <>
              <div className="barra-resultados">
                <p>
                  <strong>{resultado.frases.length}</strong> {resultado.frases.length === 1 ? "frase analisada" : "frases analisadas"} com o{" "}
                  <strong>{resultado.modelo}</strong> · decodificação {resultado.decodificacao} ·{" "}
                  {resultado.dispositivo === "webgpu" ? "GPU (WebGPU)" : "CPU (WebAssembly)"} · {resultado.tempo_ms} ms
                  {resultado.carga_ms > 500 && <> (+ {(resultado.carga_ms / 1000).toFixed(1)} s para carregar o modelo)</>}
                </p>
                <div className="acoes">
                  <button className="botao-secundario" onClick={copiarConllu}>{copiado ? "Copiado ✓" : "Copiar CoNLL-U"}</button>
                  <button className="botao-secundario" onClick={() => baixar("hubparser.conllu", resultado.conllu)}>Baixar .conllu</button>
                </div>
              </div>
              {resultado.frases.map((f, i) => (
                <Resultado key={`${resultado.tempo_ms}-${i}`} frase={f} indice={i + 1} conllu={blocos[i]} />
              ))}
            </>
          )}
          <Legenda />
        </section>
      </main>

      <footer className="rodape">
        <div className="rodape-conteudo">
          <div>
            <strong>HubParser</strong> · Hub de Inovação em Inteligência Artificial (H2IA)
            <br />Universidade Federal de Pelotas
          </div>
          <div className="rodape-links">
            <a href="https://huggingface.co/guilhermedallm4" target="_blank" rel="noreferrer">Modelos no Hugging Face</a>
            <a href="https://github.com/guilhermedallm4/HubParser" target="_blank" rel="noreferrer">Código no GitHub</a>
            <a href="https://ia.ufpel.edu.br/pt" target="_blank" rel="noreferrer">ia.ufpel.edu.br</a>
            <a href="mailto:h2ia@inf.ufpel.edu.br">h2ia@inf.ufpel.edu.br</a>
          </div>
        </div>
      </footer>
    </>
  );
}
