"""Regenera publicacao.csv a partir dos anexos, sem chamar nenhum modelo de linguagem.

    python run.py                 # usa ./anexos, escreve publicacao.csv e artefatos/
    python run.py --anexos DIR    # outra pasta com os mesmos 5 arquivos

Etapas:
 1. extrai features de todas as questões (léxico, TF-IDF, citações, árvore, assunto de origem)
 2. validação cruzada 10 dobras x 5 repetições do ranqueador na folha-verdade
 3. calibração por grupo, avaliada com validação cruzada aninhada (a calibração também é
    ajustada fora da dobra) -> tabela precisão x cobertura em artefatos/avaliacao_cv.json
 4. treina ranqueador e calibração finais com os 100 rótulos e decide as 659 questões
 5. estima precisão/cobertura esperadas nas 559 questões fora da folha (artefatos/previsao_calculada.json)
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

from classificador.decisao import (META, R_CONFIAVEL, calibrada, candidatos_decisao, decidir,
                                   grupo, meta_features, treinar_meta)
from classificador.features import NOMES, NULO
from classificador.pipeline import ajustar, alvos_treino, preparar, probs, validacao_cruzada

LAM = 1.0          # regularização L2 do ranqueador
TAU = 0.80         # limiar de confiança calibrada (meio do platô de precisão na validação aninhada)
DOBRAS, REPETICOES = 10, 5
GRADE_TAU = [0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94]

RAIZ = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(msg, flush=True)


def _pares(ext, arv, folha, oof):
    """Para cada questão rotulada: meta-features e acerto do nó específico e do tema."""
    info = {}
    for qid, (p, r) in oof.items():
        (m, cm), (t, ct) = candidatos_decisao(ext, qid, p)
        info[qid] = [(meta_features(ext, qid, m, cm, r), int(arv.correto(m, folha[qid]))),
                     (meta_features(ext, qid, t, ct, r), int(arv.correto(t, folha[qid])))]
    return info


def validacao_aninhada(ext, arv, folha, oofs):
    """Calibração ajustada fora da dobra; decisão real (decidir) avaliada em cada limiar."""
    ids = sorted(folha)
    grade = {tau: [] for tau in GRADE_TAU}
    linhas_oof = []
    for rep, oof in enumerate(oofs):
        info = _pares(ext, arv, folha, oof)
        rnd = random.Random(1000 + rep)
        ordem = ids[:]
        rnd.shuffle(ordem)
        for f in range(DOBRAS):
            teste = set(ordem[f::DOBRAS])
            pares = [pz for q in ids if q not in teste for pz in info[q]]
            wm = treinar_meta(pares)
            for q in teste:
                p, r = oof[q]
                for tau in GRADE_TAU:
                    d = decidir(ext, q, p, r, wm, tau)
                    ok = d["decisao"] == "publicar" and arv.correto(d["tema"], folha[q])
                    grade[tau].append((d["decisao"], ok, d["confianca"], q, d["grupo"]))
                if rep == 0:
                    d = decidir(ext, q, p, r, wm, TAU)
                    linhas_oof.append((q, d, r))
    tabela = []
    for tau, v in grade.items():
        pub = [x for x in v if x[0] == "publicar"]
        acertos = sum(x[1] for x in pub)
        tabela.append({"tau": tau, "cobertura": round(len(pub) / len(v), 3),
                       "precisao": round(acertos / len(pub), 3) if pub else None,
                       "confianca_media": round(sum(x[2] for x in pub) / len(pub), 3) if pub else None,
                       "publicadas_por_repeticao": round(len(pub) / len(oofs), 1),
                       "erros": dict(Counter(x[3] for x in pub if not x[1]))})
    return tabela, linhas_oof


def beta_intervalo(acertos, n, nivel=0.9, amostras=20000, semente=7):
    """Intervalo de credibilidade de Beta(1+acertos, 1+erros) por amostragem."""
    rnd = random.Random(semente)
    xs = sorted(rnd.betavariate(1 + acertos, 1 + n - acertos) for _ in range(amostras))
    a = (1 - nivel) / 2
    return xs[int(a * amostras)], xs[int((1 - a) * amostras) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anexos", default=os.path.join(RAIZ, "anexos"))
    ap.add_argument("--saida", default=os.path.join(RAIZ, "publicacao.csv"))
    ap.add_argument("--rapido", action="store_true", help="pula a validação cruzada (só gera o CSV)")
    args = ap.parse_args()
    art = os.path.join(RAIZ, "artefatos")
    os.makedirs(art, exist_ok=True)
    t0 = time.time()

    base, qs, folha, ext = preparar(args.anexos)
    arv, nos = base.arvore, base.arvore.nos
    alvos = alvos_treino(base, folha)
    log("[%.0fs] features prontas: %d questões, %d nós candidatos" % (time.time() - t0, len(qs), len(base.candidatos)))

    # ---------------------------------------------------------------- validação
    avaliacao = {}
    if not args.rapido:
        oofs = validacao_cruzada(ext, alvos, LAM, k=DOBRAS, repeticoes=REPETICOES)
        log("[%.0fs] validação cruzada do ranqueador ok" % (time.time() - t0))
        top1 = [arv.correto(candidatos_decisao(ext, q, p)[0][0], folha[q]) for oof in oofs for q, (p, r) in oof.items()]
        tabela, linhas_oof = validacao_aninhada(ext, arv, folha, oofs)
        log("[%.0fs] validação aninhada ok" % (time.time() - t0))
        for l in tabela:
            log("  tau %.2f  cobertura %.2f  precisão %s  (%s publicadas/rep.)" % (
                l["tau"], l["cobertura"], l["precisao"], l["publicadas_por_repeticao"]))
        pares_todos = [pz for oof in oofs for q, pzs in _pares(ext, arv, folha, oof).items() for pz in pzs]
        wm = treinar_meta(pares_todos, peso=1.0 / REPETICOES)
        avaliacao = {"acerto_top1_ramo": round(sum(top1) / len(top1), 3), "grade_tau": tabela,
                     "tau_escolhido": TAU}
        with open(os.path.join(art, "folha_oof.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "decisao_oof", "tema_oof", "grupo", "confianca", "r_assunto", "rotulo_central",
                        "rotulo_fora", "certo_se_publicado"])
            for q, d, r in sorted(linhas_oof):
                fv = folha[q]
                w.writerow([q, d["decisao"], nos[d["tema"]]["caminho"] if d["tema"] else "", d["grupo"],
                            "%.3f" % d["confianca"], "" if r is None else "%.2f" % r,
                            nos[fv["central"]]["caminho"] if fv["central"] else "", int(fv["fora"]),
                            int(bool(d["tema"]) and arv.correto(d["tema"], fv))])
    else:
        with open(os.path.join(art, "calibracao.json"), encoding="utf-8") as f:
            wm = json.load(f)["pesos"]

    # ---------------------------------------------------------------- modelo final
    wf, mem = ajustar(ext, alvos, LAM)
    saida = []
    for q in qs:
        qid = q["id"]
        p = probs(ext, wf, mem, qid)
        r = ext.confiabilidade(qid, mem)
        d = decidir(ext, qid, p, r, wm, TAU)
        d.update(id=qid, r=r, p_nulo=p.get(NULO, 0.0))
        saida.append(d)

    with open(args.saida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "decisao", "tema_central_id", "extras_ids", "confianca", "motivo"])
        for d in saida:
            w.writerow([d["id"], d["decisao"], d["tema"] or "", ";".join(d["extras"]),
                        "%.3f" % d["confianca"], d["motivo"]])
    with open(os.path.join(art, "decisoes_detalhadas.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lote", "na_folha", "decisao", "grupo", "confianca", "r_assunto", "p_nenhum_tema",
                    "tema_caminho", "extras_caminho", "disciplina_origem", "assunto_origem", "motivo"])
        for d, q in zip(saida, qs):
            w.writerow([d["id"], "novo" if int(d["id"][1:]) >= 360 else "banco", int(d["id"] in folha),
                        d["decisao"], d["grupo"], "%.3f" % d["confianca"],
                        "" if d["r"] is None else "%.2f" % d["r"], "%.3f" % d["p_nulo"],
                        nos[d["tema"]]["caminho"] if d["tema"] else "",
                        " | ".join(nos[x]["caminho"] for x in d["extras"]), q["disciplina"], q["assunto"], d["motivo"]])

    # ---------------------------------------------------------------- artefatos do modelo
    with open(os.path.join(art, "pesos_ranqueador.json"), "w", encoding="utf-8") as f:
        json.dump({"lambda_l2": LAM, "pesos": dict(zip(NOMES, [round(x, 4) for x in wf]))}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(art, "calibracao.json"), "w", encoding="utf-8") as f:
        json.dump({"tau": TAU, "r_confiavel": R_CONFIAVEL, "meta": META, "pesos": wm}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(art, "confiabilidade_assuntos.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["assunto_origem_normalizado", "n_questoes", "concordancia_texto", "n_rotuladas", "r", "temas_do_assunto"])
        for an, js in sorted(ext.por_assunto_todos.items(), key=lambda kv: -len(kv[1])):
            rotul = [j for j in js if j in folha]
            # r "de produção": todas as questões e rótulos do assunto
            num = sum(ext.concorda[j] for j in js)
            temas = sorted({nos[t]["titulo"] for j in js for t in ext.temas_assunto[j]})
            w.writerow([an, len(js), "%.2f" % (num / len(js)), len(rotul),
                        "%.2f" % ext.confiabilidade(js[0], mem) if len(js) > 1 else "%.2f" % ext.confiabilidade(js[0], mem),
                        "; ".join(temas)[:200]])

    # ---------------------------------------------------------------- previsão nas 559 avaliadas
    avaliadas = [d for d in saida if d["id"] not in folha]
    pub = [d for d in avaliadas if d["decisao"] == "publicar"]
    cont = Counter(d["decisao"] for d in avaliadas)
    prev = {"n_avaliadas": len(avaliadas), "decisoes": dict(cont),
            "cobertura": round(len(pub) / len(avaliadas), 3),
            "precisao_esperada_pela_calibracao": round(sum(d["confianca"] for d in pub) / max(1, len(pub)), 3),
            "por_lote": {}, "por_grupo": dict(Counter(d["grupo"] for d in pub)),
            "nivel_do_no_publicado": dict(Counter(nos[d["tema"]]["nivel"] for d in pub))}
    for lote in ("banco", "novo"):
        av = [d for d in avaliadas if ("novo" if int(d["id"][1:]) >= 360 else "banco") == lote]
        pb = [d for d in av if d["decisao"] == "publicar"]
        prev["por_lote"][lote] = {"n": len(av), "publicadas": len(pb), "cobertura": round(len(pb) / len(av), 3),
                                  "confianca_media": round(sum(d["confianca"] for d in pb) / max(1, len(pb)), 3)}
    if avaliacao:
        lt = next(l for l in avaliacao["grade_tau"] if l["tau"] == TAU)
        n_pub_rep = lt["publicadas_por_repeticao"]
        acertos = round(n_pub_rep * lt["precisao"])
        prev["validacao_folha"] = {"precisao_oof": lt["precisao"], "publicadas_por_repeticao": n_pub_rep,
                                   "intervalo_90_beta": [round(x, 3) for x in beta_intervalo(acertos, round(n_pub_rep))]}
        avaliacao["previsao"] = prev
        with open(os.path.join(art, "avaliacao_cv.json"), "w", encoding="utf-8") as f:
            json.dump(avaliacao, f, ensure_ascii=False, indent=1)
    with open(os.path.join(art, "previsao_calculada.json"), "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=1)

    log("[%.0fs] publicacao.csv escrito: %s" % (time.time() - t0, dict(Counter(d["decisao"] for d in saida))))
    log("  nas 559 avaliadas: cobertura %.3f, precisão esperada (calibração) %.3f" % (
        prev["cobertura"], prev["precisao_esperada_pela_calibracao"]))
    log("  por lote: %s" % prev["por_lote"])


if __name__ == "__main__":
    sys.exit(main())
