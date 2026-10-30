"""Treino, validação cruzada e previsão (sem nenhuma chamada a modelo de linguagem)."""
import random

from .features import NOMES, NULO, Extrator
from .kb import Base, carregar_folha, carregar_questoes
from .modelo import opcoes, prever, treinar


def preparar(pasta):
    base = Base(pasta)
    qs = carregar_questoes(pasta)
    folha = carregar_folha(pasta)
    ext = Extrator(base, qs)
    return base, qs, folha, ext


def alvos_treino(base, folha):
    """Rótulos da folha -> conjunto de nós-alvo (duplicatas mapeadas para o nó canônico)."""
    out = {}
    for qid, r in folha.items():
        if r["fora"]:
            out[qid] = {NULO}
            continue
        s = set()
        for a in [r["central"]] + r["extras"]:
            if not a:
                continue
            a = base.alias.get(a, a)
            if a in base.candidatos:
                s.add(a)
        out[qid] = s or {NULO}
    return out


def montar(ext, alvos, mem):
    dados = []
    for qid, alvo in alvos.items():
        cand, X = ext.matriz(qid, mem, extra=[a for a in alvo if a != NULO])
        pos = {i for i, c in enumerate(cand) if c in alvo}
        dados.append((X, pos))
    return dados


def ajustar(ext, alvos, lam):
    mem = ext.memoria(alvos)
    return treinar(montar(ext, alvos, mem), len(NOMES), lam=lam), mem


def probs(ext, w, mem, qid):
    cand, X = ext.matriz(qid, mem)
    return prever(w, cand, X)


def validacao_cruzada(ext, alvos, lam, k=10, repeticoes=1, semente=13):
    """Probabilidades fora-da-dobra para cada questão rotulada (lista por repetição)."""
    ids = sorted(alvos)
    saidas = []
    for r in range(repeticoes):
        rnd = random.Random(semente + r)
        ordem = ids[:]
        rnd.shuffle(ordem)
        oof = {}
        for f in range(k):
            teste = ordem[f::k]
            treino = {q: alvos[q] for q in ordem if q not in set(teste)}
            w, mem = ajustar(ext, treino, lam)
            for q in teste:
                oof[q] = (probs(ext, w, mem, q), ext.confiabilidade(q, mem))
        saidas.append(oof)
    return saidas


def resumo_oof(base, folha, oof):
    """Para cada questão: (melhor nó, conf bruta, acerto; melhor tema, conf, acerto; p_nulo)."""
    arv = base.arvore
    linhas = {}
    for qid, (p, _r) in oof.items():
        m, cm, t, ct = opcoes(arv, p)
        linhas[qid] = {"no": m, "conf_no": cm, "ok_no": arv.correto(m, folha[qid]),
                       "tema": t, "conf_tema": ct, "ok_tema": arv.correto(t, folha[qid]),
                       "p_nulo": p.get(NULO, 0.0)}
    return linhas
