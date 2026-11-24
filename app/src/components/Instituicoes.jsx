// Faixa institucional do rodapé: quem desenvolve (H2IA, PPGC, UFPel) e com quem colabora (POeTiSA).
export default function Instituicoes() {
  return (
    <div className="instituicoes">
      <div className="instituicoes-grupo">
        <span className="instituicoes-rotulo">Desenvolvimento</span>
        <div className="instituicoes-logos">
          <a href="https://ia.ufpel.edu.br/pt" target="_blank" rel="noreferrer" className="marca"
            title="H2IA: Hub de Inovação em Inteligência Artificial da UFPel">
            <img src="/h2ia-logo.png" alt="H2IA: Hub de Inovação em Inteligência Artificial" className="marca-h2ia" />
          </a>
          <a href="https://wp.ufpel.edu.br/computacao/ppgc/" target="_blank" rel="noreferrer" className="marca"
            title="PPGC: Programa de Pós-Graduação em Computação da UFPel">
            <img src="/logos/ppgc.png" alt="PPGC: Programa de Pós-Graduação em Computação da UFPel" className="marca-placa marca-ppgc" />
          </a>
          <a href="https://portal.ufpel.edu.br/" target="_blank" rel="noreferrer" className="marca"
            title="Universidade Federal de Pelotas">
            <img src="/logos/ufpel.png" alt="Universidade Federal de Pelotas (UFPel)" className="marca-ufpel" />
          </a>
        </div>
      </div>
      <div className="instituicoes-grupo">
        <span className="instituicoes-rotulo">Em colaboração com</span>
        <div className="instituicoes-logos">
          <a href="https://sites.google.com/icmc.usp.br/poetisa" target="_blank" rel="noreferrer" className="marca"
            title="POeTiSA: POrtuguese processing – Towards Syntactic Analysis and parsing">
            <img src="/logos/poetisa.png" alt="POeTiSA: POrtuguese processing – Towards Syntactic Analysis and parsing" className="marca-placa marca-poetisa" />
          </a>
        </div>
      </div>
    </div>
  );
}
