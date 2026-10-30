"""Calibração da confiança e regra de decisão publicar / revisar / fora.

A confiança final é um Platt por grupo de evidência, ajustado nas previsões fora-da-dobra
da folha-verdade:  P(certo) = σ(a_grupo + b_grupo · logit(confiança bruta do ranqueador)).

Grupos (definidos pela concordância entre o texto e o assunto de origem):
  G1 assunto_confiavel  o assunto de origem tem confiabilidade r >= R_CONFIAVEL e aponta para
                        o mesmo ramo que o sistema escolheu
  G2 sem_assunto        a questão não tem assunto de origem
  G3 demais             assunto-lixeira (r baixo) ou assunto que diverge do texto
Na folha-verdade, G1 acerta ~95% e G3 ~65%, quase independente da confiança do texto: em G3
as questões "certeiras" para o texto são justamente as que a equipe rotulou de outro jeito.
"""
import math

from .features import NULO
from .modelo import _solve, logit, opcoes

META = ["G1_assunto_confiavel", "G2_sem_assunto", "G3_demais",
        "G1_logit_conf", "G2_logit_conf", "G3_logit_conf"]
PRIOR_META = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
R_CONFIAVEL = 0.7


def apoio_assunto(ext, qid, no):
    e = ext.est[qid]
    for m in ext.arv.ramo(no):
        if e["titulo_a"].get(m, 0) > 0 or e["lex_a"].get(m, 0) > 0:
            return 1.0
    return 0.0


def grupo(ext, qid, no, r):
    if r is None:
        return 1
    if r >= R_CONFIAVEL and apoio_assunto(ext, qid, no):
        return 0
    return 2


def meta_features(ext, qid, no, conf, r):
    """r = confiabilidade do assunto de origem (None quando não há assunto)."""
    z = [0.0] * 6
    g = grupo(ext, qid, no, r)
    z[g] = 1.0
    z[3 + g] = logit(conf)
    return z


def _sig(x):
    return 1 / (1 + math.exp(-x)) if x > -30 else 0.0


def treinar_meta(pares, forca=1.0, peso=1.0):
    """pares: (z, y). Regressão logística com prior gaussiano centrado em PRIOR_META."""
    d = len(META)
    w = PRIOR_META[:]
    for _ in range(60):
        g = [-forca * (w[k] - PRIOR_META[k]) for k in range(d)]
        H = [[forca if i == j else 0.0 for j in range(d)] for i in range(d)]
        for z, y in pares:
            p = _sig(sum(a * b for a, b in zip(w, z)))
            for i in range(d):
                g[i] += peso * (y - p) * z[i]
                for j in range(d):
                    H[i][j] += peso * p * (1 - p) * z[i] * z[j]
        dw = _solve(H, g)
        w = [a + b for a, b in zip(w, dw)]
        if sum(abs(x) for x in dw) < 1e-8:
            break
    return w


def calibrada(wm, z):
    return _sig(sum(a * b for a, b in zip(wm, z)))


def candidatos_decisao(ext, qid, probs):
    """(nó mais provável, conf bruta) e (melhor tema nível 1, conf bruta)."""
    m, cm, t, ct = opcoes(ext.arv, probs)
    return (m, cm), (t, ct)


def extras(arv, probs, central, limite=0.15, maximo=2):
    ramo = arv.ramo(central)
    out = []
    for m, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        if m == NULO or m in ramo or p < limite:
            continue
        if any(m in arv.ramo(x) for x in out):
            continue
        out.append(m)
        if len(out) >= maximo:
            break
    return out


TAU_SUBTEMA = 0.90  # descer ao subtema exige mais confiança: publicar no tema também conta como certo


def decidir(ext, qid, probs, r, wm, tau, fora_nulo=0.5):
    """Aplica `_decidir` e depois as convenções da equipe (kb/convencoes.txt) ao nó escolhido."""
    d = _decidir(ext, qid, probs, r, wm, tau, fora_nulo)
    conv = ext.base.convencao
    if d["tema"] in conv:
        orig = d["tema"]
        d["tema"] = conv[orig]
        d["extras"] = [orig] + [x for x in d["extras"] if x not in (orig, d["tema"])][:1]
    return d


def _decidir(ext, qid, probs, r, wm, tau, fora_nulo=0.5):
    """-> dict(decisao, tema, extras, confianca, motivo, grupo).

    Só publica com evidência dupla (G1: texto + assunto confiável concordando) ou sem assunto
    (G2) — nunca G3: na folha-verdade, G3 não passa de ~85% de precisão em nenhum limiar.
    Entre os publicáveis, prefere o nó mais específico; se ele não passa no limiar, tenta o tema
    (nível 1), que pela regra oficial também conta como certo quando o rótulo é um subtema dele.
    """
    arv = ext.arv
    nos = arv.nos
    e = ext.est[qid]
    if e["compat"] == "FORA":
        return {"decisao": "fora", "tema": None, "extras": [], "confianca": 0.0, "grupo": "-",
                "motivo": "disciplina de origem (%s) sem ramo na árvore" % ext.qs[qid]["disciplina"]}
    (m, cm), (t, ct) = candidatos_decisao(ext, qid, probs)
    gm, gt = grupo(ext, qid, m, r), grupo(ext, qid, t, r)
    cal_m = calibrada(wm, meta_features(ext, qid, m, cm, r))
    cal_t = calibrada(wm, meta_features(ext, qid, t, ct, r))
    p_nulo = probs.get(NULO, 0.0)
    if p_nulo >= fora_nulo and p_nulo > max(cm, ct):
        return {"decisao": "fora", "tema": None, "extras": [], "confianca": round(1 - p_nulo, 3),
                "grupo": "-", "motivo": "nenhum tema com evidência suficiente (P(nenhum tema) = %.2f)" % p_nulo}
    tau_m = max(tau, TAU_SUBTEMA) if nos[m]["nivel"] == 2 else tau
    if gm != 2 and cal_m >= tau_m:
        return {"decisao": "publicar", "tema": m, "extras": extras(arv, probs, m),
                "confianca": cal_m, "grupo": "G%d" % (gm + 1), "motivo": ""}
    if gt != 2 and cal_t >= tau:
        return {"decisao": "publicar", "tema": t, "extras": extras(arv, probs, t),
                "confianca": cal_t, "grupo": "G%d" % (gt + 1), "motivo": ""}
    alt = [x for x in sorted(probs, key=lambda k: -probs[k]) if x != NULO and x not in arv.ramo(m)][:1]
    if gm == 2 and gt == 2:
        if r is not None and r < R_CONFIAVEL:
            motivo = "assunto de origem pouco confiável ('%s', r=%.2f)" % (ext.qs[qid]["assunto"][:50], r)
        else:
            motivo = "assunto de origem ('%s') aponta para outro ramo" % ext.qs[qid]["assunto"][:50]
    else:
        motivo = "confiança %.2f abaixo do limiar %.2f" % (cal_m, tau)
    if alt:
        motivo += "; alternativa: %s" % nos[alt[0]]["caminho"]
    return {"decisao": "revisar", "tema": m, "extras": extras(arv, probs, m),
            "confianca": cal_m, "grupo": "G%d" % (gm + 1), "motivo": motivo}
