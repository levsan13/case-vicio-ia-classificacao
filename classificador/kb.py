"""Carrega anexos (questões, árvore, folha-verdade) e a base de conhecimento em kb/."""
import csv
import json
import os
import re

from .texto import chave_artigo, limpar_html, normalizar, tokens

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _linhas_kb(nome):
    with open(os.path.join(RAIZ, "kb", nome), encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                yield [c.strip() for c in ln.split("|")]


# ---------------------------------------------------------------- anexos

def carregar_questoes(pasta):
    qs = []
    with open(os.path.join(pasta, "questoes.jsonl"), encoding="utf-8") as f:
        for ln in f:
            q = json.loads(ln)
            alts = q.get("alternativas") or {}
            q["enunciado"] = limpar_html(q.get("enunciado_html", ""))
            q["texto"] = "\n".join([q["enunciado"], q.get("comando") or ""]
                                   + [limpar_html(v) for v in alts.values()])
            q["assunto"] = (q.get("assunto_origem") or "").strip()
            q["disciplina"] = (q.get("disciplina_origem") or "").strip()
            qs.append(q)
    return qs


def carregar_folha(pasta):
    out = {}
    with open(os.path.join(pasta, "folha_verdade.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            extras = [e for e in (r["extras_ids"] or "").split(";") if e]
            out[r["id"]] = {"central": r["tema_central_id"] or None, "extras": extras,
                            "fora": r["fora_da_taxonomia"] == "1"}
    return out


class Arvore:
    def __init__(self, pasta):
        self.nos = {}
        with open(os.path.join(pasta, "temas.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = re.search(r"(\d+) quest", r["descricao"] or "")
                self.nos[r["tema_id"]] = {
                    "id": r["tema_id"], "pai": r["parent_id"] or None, "nivel": int(r["nivel"]),
                    "titulo": r["titulo"], "caminho": r["caminho"], "disciplina": r["disciplina"],
                    "ativo": r["status"] == "active", "n_source": int(m.group(1)) if m else None,
                    "filhos": []}
        for n in self.nos.values():
            if n["pai"] in self.nos:
                self.nos[n["pai"]]["filhos"].append(n["id"])

    def ramo(self, tid):
        """Nós em que publicar conta como certo se o rótulo for `tid`: ele, o pai e os filhos."""
        n = self.nos.get(tid)
        if not n:
            return set()
        s = {tid} | set(n["filhos"])
        if n["pai"]:
            s.add(n["pai"])
        return s

    def correto(self, publicado, rotulo):
        """Regra oficial: central, extra, ou pai/filho de um deles."""
        if rotulo is None or rotulo["fora"] or not publicado:
            return False
        alvos = [rotulo["central"]] + rotulo["extras"]
        return any(publicado in self.ramo(a) for a in alvos if a)


# ---------------------------------------------------------------- base de conhecimento

def _parse_dispositivo(s):
    """'CP:155-156' -> ('CP', lo, hi); 'CPP:28A' -> ('CPP', k, k); 'CP:121' -> exato."""
    dip, _, arts = s.partition(":")
    dip = dip.strip().upper()
    arts = arts.strip().upper()
    m = re.fullmatch(r"(\d+)([A-Z]?)(?:-(\d+)([A-Z]?))?", arts)
    if not m:
        raise ValueError("dispositivo inválido: %r" % s)
    a, sa, b, sb = m.groups()
    lo = chave_artigo(a, sa.lower())
    if b is None:
        hi = lo
    else:
        hi = chave_artigo(b, sb.lower()) if sb else chave_artigo(b) + 99
    return dip, lo, hi, 1 if b is None else max(1, int(b) - int(a) + 1)


class Base:
    """Árvore + léxico + dispositivos + nós especiais + mapa de disciplinas."""

    def __init__(self, pasta_anexos):
        self.arvore = Arvore(pasta_anexos)
        nos = self.arvore.nos
        self.alias = {}
        for tid, canon, _motivo in _linhas_kb("nos_especiais.txt"):
            self.alias[tid] = canon
        self.termos = {tid: [] for tid in nos}
        self.dispositivos = {tid: [] for tid in nos}
        for cols in _linhas_kb("lexico_temas.txt"):
            tid, _titulo, termos = cols[0], cols[1], cols[2]
            disp = cols[3] if len(cols) > 3 else ""
            if tid not in nos:
                raise ValueError("tema_id do léxico não existe em temas.csv: %s" % tid)
            self.termos[tid] = [t.strip() for t in termos.split(";") if t.strip()]
            self.dispositivos[tid] = [_parse_dispositivo(d) for d in disp.split(";") if d.strip()]
        self.convencao = {de: para for de, para, _ev in _linhas_kb("convencoes.txt")}
        self.disc_map = {}
        for cols in _linhas_kb("disciplinas.txt"):
            prim = [x.strip() for x in cols[1].split(";") if x.strip()]
            sec = [x.strip() for x in cols[2].split(";") if x.strip()] if len(cols) > 2 else []
            self.disc_map[normalizar(cols[0])] = (prim, sec)
        # candidatos a publicação: ativos, não-raiz, não-duplicata
        self.candidatos = [tid for tid, n in nos.items()
                           if n["ativo"] and n["nivel"] > 0 and tid not in self.alias]
        # frases (tuplas de radicais) por nó: título + termos do léxico
        self.frases = {}
        for tid in self.candidatos:
            fr = set()
            tit = tokens(re.sub(r"^\d+\s+", "", nos[tid]["titulo"]))
            if tit:
                fr.add(tuple(tit))
            for t in self.termos[tid]:
                tk = tokens(t)
                if tk:
                    fr.add(tuple(tk))
            self.frases[tid] = fr

    def disciplina_compat(self, disciplina_origem):
        """-> (dict disciplina_arvore -> peso) ou None se desconhecida; 'FORA' se sem ramo."""
        m = self.disc_map.get(normalizar(disciplina_origem or ""))
        if m is None:
            return None
        prim, sec = m
        if prim == ["FORA"]:
            return "FORA"
        d = {x: 1.0 for x in prim}
        for x in sec:
            d.setdefault(x, 0.5)
        return d
