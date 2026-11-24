import { useEffect, useMemo, useRef, useState } from "react";
import Citacao from "./components/Citacao.jsx";
import Contador from "./components/Contador.jsx";
import Instituicoes from "./components/Instituicoes.jsx";
import Resultado from "./components/Resultado.jsx";
import Progresso from "./components/Progresso.jsx";
import SeletorDecodificacao from "./components/SeletorDecodificacao.jsx";
import SeletorDispositivo from "./components/SeletorDispositivo.jsx";
import SeletorModelo from "./components/SeletorModelo.jsx";
import * as motor from "./motor/cliente.js";
import { DECODIFICACOES, LIMITES, MODELOS } from "./modelos.js";
import { EXEMPLOS } from "./rotulos.js";
import { baixar, blocosConllu } from "./util.js";

// Interface simples (padrão, para quem não é da área) ou completa (escolha de modelo, decodificação e processador).
const RECOMENDADO = { modelo: "HubParser-Base-B", decod: "eisner" };
function modoInicial() {
  try {
    const q = new URLSearchParams(location.search).get("modo");
    if (q === "simples" || q === "completo") return q;
    return localStorage.getItem("hubparser-modo") === "completo" ? "completo" : "simples";
  } catch {
    return "simples";
  }
}

export default function App() {
  const [modo, setModo] = useState(modoInicial);
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

  function trocarModo(novo) {
    setModo(novo);
    try {
      localStorage.setItem("hubparser-modo", novo);
      const url = new URL(location.href);
      url.searchParams.set("modo", novo);
      history.replaceState(null, "", url);
    } catch { /* sem armazenamento: vale só nesta visita */ }
  }
  const simples = modo === "simples";
  // no modo simples valem sempre as opções recomendadas
  const modeloUsado = simples ? RECOMENDADO.modelo : modelo;
  const decodUsada = simples ? RECOMENDADO.decod : decod;
  const dispositivoUsado = simples ? (ambiente?.gpu?.disponivel === false ? "wasm" : dispositivo) : dispositivo;

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
      motor.preprocessar(texto, modeloUsado)
        .then((r) => valido && setPrevia(r))
        .catch(() => {})
        .finally(() => valido && setContando(false));
    }, 300);
    return () => {
      valido = false;
      clearTimeout(t);
    };
  }, [texto, modeloUsado]);

  const infoModelo = useMemo(() => MODELOS.find((m) => m.id === modeloUsado), [modeloUsado]);
  const podeAnalisar = !!texto.trim() && !analisando && (previa ? previa.ok : true);

  async function analisar() {
    if (!podeAnalisar) return;
    setAnalisando(true);
    setErro(null);
    setProgresso(null);
    try {
      const r = await motor.analisar(texto, modeloUsado, decodUsada, dispositivoUsado, setProgresso);
      setResultado(r);
      setAmbiente((a) => a && { gpu: r.gpu, baixados: a.baixados.includes(modeloUsado) ? a.baixados : [...a.baixados, modeloUsado] });
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

  const botaoAnalisar = (
    <button className="botao-principal" onClick={analisar} disabled={!podeAnalisar} aria-busy={analisando}>
      {analisando ? <><span className="spinner" aria-hidden="true" /> {progresso?.etapa === "baixando" ? "Baixando o modelo…" : progresso?.etapa === "preparando" ? "Preparando o modelo…" : "Analisando…"}</> : "Analisar"}
    </button>
  );
  const avisoDownload = !analisando && infoModelo && !ambiente?.baixados.includes(modeloUsado) && (
    <p className="ajuda centro">Na primeira análise, o navegador baixa o modelo ({infoModelo.mb} MB). Nas próximas, ele já estará guardado.</p>
  );

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
            <p>Análise de Dependência Sintática para o português</p>
          </div>
          <div className="modo" role="group" aria-label="Tipo de interface">
            <button className={simples ? "modo-botao ativo" : "modo-botao"} aria-pressed={simples} onClick={() => trocarModo("simples")}
              title="Interface simples, já configurada com as opções recomendadas">Simples</button>
            <button className={!simples ? "modo-botao ativo" : "modo-botao"} aria-pressed={!simples} onClick={() => trocarModo("completo")}
              title="Escolha do modelo, da decodificação e do processamento">Avançado</button>
          </div>
        </div>
      </header>

      <main id="conteudo" className="conteudo">
        <section className="intro" aria-labelledby="titulo-intro">
          <h2 id="titulo-intro">O que é o HubParser?</h2>
          <p>
            O HubParser é um <strong>analisador sintático automático</strong> (<em>parser</em>) do português brasileiro. Para
            cada sentença de entrada, ele produz a sua <strong>árvore sintática de dependências</strong>: indica de qual palavra
            cada palavra depende e qual função ela cumpre na frase (sujeito, objeto, adjunto…). Também identifica a{" "}
            <strong>classe gramatical</strong>, o <strong>lema</strong> e os <strong>traços morfológicos</strong> (gênero,
            número, tempo, modo…) de cada palavra, no padrão{" "}
            <a href="https://universaldependencies.org/" target="_blank" rel="noreferrer">Universal Dependencies</a>. Foi
            desenvolvido no PPGC e no H2IA da UFPel e treinado no Porttinari, em parceria com o projeto{" "}
            <a href="https://sites.google.com/icmc.usp.br/poetisa" target="_blank" rel="noreferrer">POeTiSA</a>.
          </p>
          <div className="desempenho" aria-label="Desempenho do modelo recomendado no teste do Porttinari">
            <div><strong>99,3%</strong><span>classe gramatical</span></div>
            <div><strong>99,1%</strong><span>lema</span></div>
            <div><strong>99,1%</strong><span>traços morfológicos</span></div>
            <div><strong>97,0%</strong><span>palavra-núcleo (UAS)</span></div>
            <div><strong>95,9%</strong><span>núcleo e relação (LAS)</span></div>
          </div>
          <p className="desempenho-nota">
            Acertos do modelo recomendado nas 1.683 frases de teste do Porttinari (33.580 palavras), que não foram usadas no
            treino (<a href="#como-citar">Lima, 2026</a>).
          </p>
          <p className="privacidade">
            A análise roda <strong>no seu navegador</strong>, na placa de vídeo (WebGPU) ou no
            processador. O texto não é enviado a nenhum servidor; só o modelo é baixado na primeira vez e fica guardado.
          </p>
        </section>

        {erroConfig && <div className="aviso erro" role="alert">{erroConfig}</div>}

        <section className={simples ? "cartao entrada entrada-simples" : "cartao entrada"} aria-labelledby="titulo-entrada">
          <div className="entrada-texto">
            <label id="titulo-entrada" htmlFor="texto" className="rotulo">{simples ? "Digite ou cole um texto em português" : "Texto"}</label>
            <textarea
              id="texto"
              value={texto}
              onChange={(e) => setTexto(e.target.value)}
              onKeyDown={aoTeclar}
              rows={simples ? 4 : 6}
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
            <div id="contador">{infoModelo && <Contador previa={previa} modelo={infoModelo} carregando={contando} compacto={simples} />}</div>
          </div>

          {simples ? (
            <div className="entrada-acao">
              {botaoAnalisar}
              {analisando && <Progresso progresso={progresso} modelo={infoModelo} />}
              {avisoDownload}
              <p id="ajuda-texto" className="ajuda centro">
                Já configurado com as opções recomendadas.{" "}
                <button type="button" className="link" onClick={() => trocarModo("completo")}>Ver opções avançadas</button>
              </p>
            </div>
          ) : (
            <div className="entrada-opcoes">
              <SeletorModelo modelos={MODELOS} valor={modelo} onChange={setModelo} desabilitado={analisando}
                baixados={ambiente?.baixados ?? []} />
              <SeletorDecodificacao opcoes={DECODIFICACOES} valor={decod} onChange={setDecod} desabilitado={analisando} />
              <SeletorDispositivo gpu={ambiente?.gpu} valor={dispositivo} onChange={setDispositivo} desabilitado={analisando || !ambiente} />
              {botaoAnalisar}
              {analisando && <Progresso progresso={progresso} modelo={infoModelo} />}
              {avisoDownload}
              <p id="ajuda-texto" className="ajuda centro">Atalho: <kbd>Ctrl</kbd> + <kbd>Enter</kbd></p>
            </div>
          )}
        </section>

        {erro && <div className="aviso erro" role="alert">{erro}</div>}
        {resultado?.aviso && <div className="aviso alerta" role="status">{resultado.aviso}</div>}

        <section ref={resultadosRef} className="resultados" aria-live="polite" aria-label="Resultados">
          {resultado && (
            <>
              <div className="barra-resultados">
                {simples ? (
                  <p><strong>{resultado.frases.length}</strong> {resultado.frases.length === 1 ? "frase analisada" : "frases analisadas"} em {(resultado.tempo_ms / 1000).toFixed(1).replace(".", ",")} s</p>
                ) : (
                  <p>
                    <strong>{resultado.frases.length}</strong> {resultado.frases.length === 1 ? "frase analisada" : "frases analisadas"} com o{" "}
                    <strong>{resultado.modelo}</strong> · decodificação {resultado.decodificacao} ·{" "}
                    {resultado.dispositivo === "webgpu" ? "GPU (WebGPU)" : "CPU (WebAssembly)"} · {resultado.tempo_ms} ms
                    {resultado.carga_ms > 500 && <> (+ {(resultado.carga_ms / 1000).toFixed(1)} s para carregar o modelo)</>}
                  </p>
                )}
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
          <Citacao />
        </section>
      </main>

      <footer className="rodape">
        <div className="rodape-conteudo">
          <Instituicoes />
          <div className="rodape-base">
            <div>
              <strong>HubParser</strong> · Hub de Inovação em Inteligência Artificial (H2IA) · Programa de Pós-Graduação em
              Computação (PPGC) · Universidade Federal de Pelotas
            </div>
            <div className="rodape-links">
              <a href="https://huggingface.co/H2IA" target="_blank" rel="noreferrer">Modelos no Hugging Face</a>
              <a href="https://github.com/guilhermedallm4/HubParser" target="_blank" rel="noreferrer">Código no GitHub</a>
              <a href="https://sites.google.com/icmc.usp.br/poetisa" target="_blank" rel="noreferrer">Projeto POeTiSA</a>
              <a href="mailto:h2ia@inf.ufpel.edu.br">h2ia@inf.ufpel.edu.br</a>
            </div>
          </div>
        </div>
      </footer>
    </>
  );
}
