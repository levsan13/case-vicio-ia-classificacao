"""Features de cada par (questão, nó candidato).

Cada nó da árvore recebe evidências independentes, todas interpretáveis:
  lex_q       termos do léxico do nó encontrados no texto da questão (ponderados por especificidade)
  lex_a_*     idem no `assunto_origem`, separado pela confiabilidade r do assunto (ver abaixo):
              lex_a_conf = lex_a * r e lex_a_lix = lex_a * (1 - r)
  titulo_a    o título do nó aparece inteiro no assunto de origem (vezes r)
  cos_q/cos_a similaridade TF-IDF entre texto/assunto e a "ficha" do nó (título + pai + léxico)
  art_exp     artigo citado com diploma explícito cai nos dispositivos do nó ("art. 155 do CP")
  art_inf     artigo citado sem diploma, com o diploma inferido pelo contexto
  lei         número de lei citado pertence ao nó (Lei 11.343 -> Tráfico de Drogas)
  disc        compatibilidade entre disciplina de origem e o ramo do nó (1 / 0,5 / 0)
  *_filho/pai as mesmas evidências propagadas na árvore (tema herda do melhor subtema e vice-versa)
  nivel1      o nó é um tema (nível 1), não um subtema
  knn_q/knn_a vizinhos rotulados da folha-verdade (texto parecido / mesmo assunto de origem)
  lex_f/cos_f a "frase-tema" que a própria questão anuncia ("No que se refere ao Tribunal do Júri")
  geral_lex   lex_q só para nós da parte geral do CP (arts. 1-120): a equipe rotula pelo
              conceito testado (erro de tipo, tentativa) e não pelo crime do enunciado
e um candidato especial NULO ("nenhum tema serve") com viés próprio.

Confiabilidade do assunto de origem (detector de "lixeiras" do site de origem): para cada
assunto, a fração das OUTRAS questões com o mesmo assunto em que ele concorda com o texto
(sem rótulo, sobre as 659 questões) e com o rótulo da equipe (só questões da folha usadas no
treino, peso 3), suavizada para a média da classe de frequência do assunto (assuntos únicos
concordam ~75% das vezes; os que aparecem 10+ vezes, ~35%). "Concurso de Crimes" sai perto de 0,1;
"Lei Maria da Penha - Lei nº 11.340" perto de 0,9.
"""
import math
import re
from collections import Counter, defaultdict

from .texto import (diplomas_mencionados, extrair_artigos, extrair_leis, frase_tema,
                    ngramas, normalizar, tokens)

NOMES = ["lex_q", "lex_a_conf", "lex_a_lix", "titulo_a", "cos_q", "cos_a", "art_exp",
         "art_inf", "lei", "disc", "lex_filho", "lex_pai", "cos_filho", "nivel1",
         "knn_q", "knn_a", "lex_f", "cos_f", "geral_lex",
         "nulo", "nulo_disc", "nulo_knn", "nulo_knn_a"]
IDX = {n: i for i, n in enumerate(NOMES)}
NULO = "NULO"
TOPK = 40
ALFA_ASSUNTO = 2.0   # pseudo-contagens da média geral na confiabilidade do assunto
PESO_ROTULO = 3.0    # uma questão rotulada vale 3 não rotuladas na confiabilidade do assunto


def _vetor(toks, idf):
    tf = Counter(toks)
    v = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in tf.items()}
    nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / nrm for t, x in v.items() if x > 0}


def _classe_freq(n):
    return "1" if n == 1 else "2-3" if n <= 3 else "4-9" if n <= 9 else "10+"


def _cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


class Extrator:
    def __init__(self, base, questoes):
        self.base = base
        self.arv = base.arvore
        self.qs = {q["id"]: q for q in questoes}
        cands = base.candidatos
        self.cands = cands
        nos = self.arv.nos

        # IDF sobre as questões (corpus inteiro, sem rótulos)
        toks_q = {qid: tokens(q["texto"]) for qid, q in self.qs.items()}
        df = Counter()
        for tk in toks_q.values():
            df.update(set(tk))
        N = len(toks_q)
        self.idf = {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}
        idf_max = math.log(N + 1) + 1

        # ficha de cada nó: título x2, título do pai, termos do léxico
        self.vnode = {}
        for tid in cands:
            n = nos[tid]
            tk = tokens(n["titulo"]) * 2
            if n["pai"] and nos[n["pai"]]["nivel"] > 0:
                tk += tokens(nos[n["pai"]]["titulo"])
            for t in base.termos[tid]:
                tk += tokens(t)
            for t in tk:
                self.idf.setdefault(t, idf_max)
            self.vnode[tid] = _vetor(tk, self.idf)

        # especificidade das frases do léxico: frase presente em muitos nós vale menos
        dff = Counter()
        for tid in cands:
            dff.update(base.frases[tid])
        nc = len(cands)
        self.peso_frase = {f: (math.log((nc + 1) / (c + 1)) + 1) * (1 + 0.5 * (len(f) - 1))
                           for f, c in dff.items()}

        # nós da parte geral do Código Penal (dispositivos dentro dos arts. 1-120), herdado pelos filhos
        self.geral = {}
        for tid in cands:
            ds = base.dispositivos[tid] or (base.dispositivos.get(nos[tid]["pai"]) or [])
            self.geral[tid] = any(d == "CP" and hi < 12100 for d, lo, hi, _ in ds)
        self.tit = {tid: tuple(tokens(re.sub(r"^\d+\s+", "", nos[tid]["titulo"]))) for tid in cands}
        self.vq = {qid: _vetor(tk, self.idf) for qid, tk in toks_q.items()}
        self.est = {qid: self._estaticas(q, toks_q[qid]) for qid, q in self.qs.items()}
        self._concordancia_assuntos()

    # ------------------------------------------------------------ evidências estáticas
    def _lex(self, grams, tid):
        return sum(self.peso_frase[f] for f in self.base.frases[tid] if f in grams)

    def _estaticas(self, q, tq):
        base, nos = self.base, self.arv.nos
        gq = ngramas(tq)
        # "(Parte 02)", "Parte 03" são paginação do site de origem, não assunto
        ta = tokens(re.sub(r"(?i)\(?\s*parte\s*0*\d+\s*\)?", " ", q["assunto"]))
        ga = ngramas(ta)
        va = _vetor(ta, self.idf) if ta else {}
        an = normalizar(q["assunto"])

        texto_n = normalizar(q["texto"])
        assunto_n = normalizar(q["assunto"])
        cites = extrair_artigos(texto_n)
        leis = extrair_leis(texto_n) | extrair_leis(assunto_n)
        mencionados = diplomas_mencionados(texto_n)
        compat = base.disciplina_compat(q["disciplina"])
        dip_ctx = None
        codigos = mencionados & {"CP", "CPP"}
        if len(codigos) == 1:
            dip_ctx = next(iter(codigos))
        elif isinstance(compat, dict):
            if compat.get("Processo Penal") == 1.0:
                dip_ctx = "CPP"
            elif compat.get("Penal") == 1.0:
                dip_ctx = "CP"
        cites_inf = [(dip_ctx, k) for d, k, exp in cites if d is None and dip_ctx]
        cites_exp = [(d, k) for d, k, exp in cites if d is not None]

        ft = tokens(frase_tema(q["enunciado"], q.get("comando") or ""))
        gf = ngramas(ft)
        vf = _vetor(ft, self.idf) if ft else {}
        e = {"lex_q": {}, "lex_a": {}, "titulo_a": {}, "cos_q": {}, "cos_a": {},
             "art_exp": {}, "art_inf": {}, "lei": {}, "disc": {}, "lex_f": {}, "cos_f": {}}
        vq = self.vq[q["id"]]
        for tid in self.cands:
            e["lex_q"][tid] = math.log1p(self._lex(gq, tid))
            if ga:
                e["lex_a"][tid] = math.log1p(self._lex(ga, tid))
                tit = self.tit[tid]
                e["titulo_a"][tid] = 1.0 if tit and tit in ga else 0.0
                e["cos_a"][tid] = _cos(va, self.vnode[tid])
            e["cos_q"][tid] = _cos(vq, self.vnode[tid])
            if ft:
                e["lex_f"][tid] = math.log1p(self._lex(gf, tid))
                e["cos_f"][tid] = _cos(vf, self.vnode[tid])
            ax = ai = lz = 0.0
            for dip, lo, hi, larg in base.dispositivos[tid]:
                esp = 1.0 / (1.0 + math.log(larg))
                for d, k in cites_exp:
                    if d == dip and lo <= k <= hi:
                        ax = max(ax, esp)
                for d, k in cites_inf:
                    if d == dip and lo <= k <= hi:
                        ai = max(ai, esp)
                if dip in leis:
                    lz = 1.0
            e["art_exp"][tid], e["art_inf"][tid], e["lei"][tid] = ax, ai, lz
            if isinstance(compat, dict):
                e["disc"][tid] = compat.get(nos[tid]["disciplina"], 0.0)
            else:
                e["disc"][tid] = 0.5
        # pré-ranking só com evidência estática (para limitar os candidatos do modelo)
        pre = {}
        for tid in self.cands:
            pre[tid] = (e["lex_q"][tid] + 1.5 * e["lex_a"].get(tid, 0) + 3 * e["cos_q"][tid]
                        + 2 * e["cos_a"].get(tid, 0) + 2 * (e["art_exp"][tid] + e["art_inf"][tid])
                        + e["lei"][tid] + 0.5 * e["disc"][tid] + e["lex_f"].get(tid, 0)
                        + 2 * e["cos_f"].get(tid, 0))
        e["top"] = sorted(self.cands, key=lambda t: -pre[t])[:TOPK]
        e["assunto_norm"] = an
        e["compat"] = compat
        e["leis"] = sorted(leis)
        e["cites"] = cites_exp + cites_inf
        return e

    # ------------------------------------------------------------ confiabilidade do assunto
    def tema_de(self, tid):
        n = self.arv.nos[tid]
        return tid if n["nivel"] == 1 else n["pai"]

    def _concordancia_assuntos(self):
        """Sem rótulos: o assunto de origem aponta para um dos 2 temas que o texto sozinho indica?"""
        self.temas_assunto = {}
        self.concorda = {}
        self.por_assunto_todos = defaultdict(list)
        for qid, e in self.est.items():
            if not e["assunto_norm"]:
                continue
            self.por_assunto_todos[e["assunto_norm"]].append(qid)
            ta = {self.tema_de(n) for n in self.cands
                  if e["titulo_a"].get(n, 0) > 0 or e["lex_a"].get(n, 0) > 0}
            self.temas_assunto[qid] = ta
            sc = defaultdict(float)
            for n in self.cands:
                s = (e["lex_q"][n] + 3 * e["cos_q"][n] + 2 * (e["art_exp"][n] + e["art_inf"][n])
                     + e["lei"][n] + e["lex_f"].get(n, 0) + 2 * e["cos_f"].get(n, 0))
                t = self.tema_de(n)
                sc[t] = max(sc[t], s)
            top2 = sorted(sc, key=lambda t: -sc[t])[:2]
            self.concorda[qid] = 1.0 if ta & set(top2) else 0.0
        por_classe = defaultdict(list)
        for an, js in self.por_assunto_todos.items():
            for j in js:
                por_classe[_classe_freq(len(js))].append(self.concorda[j])
        self.prior_classe = {c: sum(v) / len(v) for c, v in por_classe.items()}

    def confiabilidade(self, qid, mem=None):
        """r em [0, 1] do assunto de origem desta questão (None se não há assunto)."""
        an = self.est[qid]["assunto_norm"]
        if not an:
            return None
        prior = self.prior_classe[_classe_freq(len(self.por_assunto_todos[an]))]
        num = ALFA_ASSUNTO * prior
        den = ALFA_ASSUNTO
        alvos = mem["alvos"] if mem else {}
        for j in self.por_assunto_todos[an]:
            if j == qid:
                continue
            num += self.concorda[j]
            den += 1
            if j in alvos:
                num += PESO_ROTULO * (1.0 if self.temas_assunto[j] & alvos[j] else 0.0)
                den += PESO_ROTULO
        return num / den

    # ------------------------------------------------------------ vizinhos rotulados
    def memoria(self, rotulos):
        """rotulos: {qid: conjunto de nós (já com alias) ou {NULO}} -> estrutura para knn."""
        nos = self.arv.nos
        mem = {}
        for qid, alvos in rotulos.items():
            exp = set()
            for a in alvos:
                exp.add(a)
                if a != NULO and nos[a]["pai"] and nos[nos[a]["pai"]]["nivel"] > 0:
                    exp.add(nos[a]["pai"])
            mem[qid] = exp
        por_assunto = defaultdict(list)
        for qid in mem:
            an = self.est[qid]["assunto_norm"]
            if an:
                por_assunto[an].append(qid)
        return {"alvos": mem, "por_assunto": por_assunto}

    def _knn(self, qid, mem):
        kq = defaultdict(float)
        ka = defaultdict(float)
        if not mem:
            return kq, ka
        vq = self.vq[qid]
        for j, alvos in mem["alvos"].items():
            if j == qid:
                continue
            s = _cos(vq, self.vq[j])
            for a in alvos:
                if s > kq[a]:
                    kq[a] = s
        an = self.est[qid]["assunto_norm"]
        viz = [j for j in mem["por_assunto"].get(an, []) if j != qid] if an else []
        if viz:
            cont = Counter()
            for j in viz:
                cont.update(mem["alvos"][j])
            for a, c in cont.items():
                ka[a] = c / (len(viz) + 1)
        return kq, ka

    # ------------------------------------------------------------ matriz de um questão
    def matriz(self, qid, mem=None, extra=()):
        """-> (lista de candidatos incluindo NULO, lista de vetores de features)."""
        e = self.est[qid]
        nos = self.arv.nos
        kq, ka = self._knn(qid, mem)
        cand = list(e["top"])
        vistos = set(cand)
        for a in list(kq) + list(ka) + list(extra):
            if a != NULO and a not in vistos and a in self.vnode and (kq.get(a, 0) > 0.25 or ka.get(a, 0) > 0 or a in extra):
                cand.append(a)
                vistos.add(a)
        linhas = []
        r = self.confiabilidade(qid, mem)
        r = 0.0 if r is None else r
        for tid in cand:
            n = nos[tid]
            filhos = [c for c in n["filhos"] if c in self.vnode]
            pai = n["pai"] if n["pai"] in self.vnode else None
            la = e["lex_a"].get(tid, 0.0)
            x = [0.0] * len(NOMES)
            x[0] = e["lex_q"][tid]
            x[1] = la * r
            x[2] = la * (1 - r)
            x[3] = e["titulo_a"].get(tid, 0.0) * r
            x[4] = e["cos_q"][tid]
            x[5] = e["cos_a"].get(tid, 0.0)
            x[6] = e["art_exp"][tid]
            x[7] = e["art_inf"][tid]
            x[8] = e["lei"][tid]
            x[9] = e["disc"][tid]
            x[10] = max((e["lex_q"][c] for c in filhos), default=0.0)
            x[11] = e["lex_q"][pai] if pai else 0.0
            x[12] = max((e["cos_q"][c] for c in filhos), default=0.0)
            x[13] = 1.0 if n["nivel"] == 1 else 0.0
            x[14] = kq.get(tid, 0.0)
            x[15] = ka.get(tid, 0.0)
            x[16] = e["lex_f"].get(tid, 0.0)
            x[17] = e["cos_f"].get(tid, 0.0)
            x[18] = e["lex_q"][tid] if self.geral[tid] else 0.0
            linhas.append(x)
        xn = [0.0] * len(NOMES)
        xn[IDX["nulo"]] = 1.0
        xn[IDX["nulo_disc"]] = 1.0 if e["compat"] is None else 0.0
        xn[IDX["nulo_knn"]] = kq.get(NULO, 0.0)
        xn[IDX["nulo_knn_a"]] = ka.get(NULO, 0.0)
        cand.append(NULO)
        linhas.append(xn)
        return cand, linhas
