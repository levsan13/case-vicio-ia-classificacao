"""Ranqueador linear sobre os nós candidatos (softmax condicional) + calibração de Platt.

Para cada questão, score(nó) = w · features(questão, nó) e P(nó | questão) = softmax dos scores
entre os candidatos (incluindo NULO). O alvo de treino é o conjunto {central} ∪ extras da
folha-verdade (qualquer um deles serve), otimizado por EM generalizado com passos de Newton.
São ~19 pesos: pouco parâmetro para 100 rótulos, e cada peso é legível.
"""
import math

from .features import NULO


def _solve(A, b):
    """Resolve A x = b (A simétrica positiva definida, d pequeno) por eliminação de Gauss."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-12:
            continue
        for r in range(c + 1, n):
            f = M[r][c] / d
            if f:
                Mr, Mc = M[r], M[c]
                for k in range(c, n + 1):
                    Mr[k] -= f * Mc[k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))
        x[r] = s / M[r][r] if abs(M[r][r]) > 1e-12 else 0.0
    return x


def softmax(scores):
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    z = sum(e)
    return [v / z for v in e]


def _loglik(w, dados, lam):
    ll = -0.5 * lam * sum(v * v for v in w)
    for X, pos in dados:
        p = softmax([sum(a * b for a, b in zip(w, x)) for x in X])
        ll += math.log(max(1e-300, sum(p[i] for i in pos)))
    return ll


def treinar(dados, d, lam=1.0, iters=40):
    """dados: lista de (X: lista de vetores, pos: conjunto de índices corretos)."""
    w = [0.0] * d
    ll = _loglik(w, dados, lam)
    for _ in range(iters):
        g = [-lam * v for v in w]
        H = [[lam if i == j else 0.0 for j in range(d)] for i in range(d)]  # -Hessiana
        for X, pos in dados:
            p = softmax([sum(a * b for a, b in zip(w, x)) for x in X])
            pp = sum(p[i] for i in pos) or 1e-300
            mu = [0.0] * d
            for i, x in enumerate(X):
                coef = (p[i] / pp if i in pos else 0.0) - p[i]
                pi = p[i]
                nz = [(k, v) for k, v in enumerate(x) if v]
                for k, v in nz:
                    g[k] += coef * v
                    mu[k] += pi * v
                    Hk = H[k]
                    pv = pi * v
                    for k2, v2 in nz:
                        Hk[k2] += pv * v2
            for k in range(d):
                if mu[k]:
                    Hk = H[k]
                    for k2 in range(d):
                        Hk[k2] -= mu[k] * mu[k2]
        delta = _solve(H, g)
        passo = 1.0
        while passo > 1e-3:
            w2 = [a + passo * b for a, b in zip(w, delta)]
            ll2 = _loglik(w2, dados, lam)
            if ll2 >= ll - 1e-9:
                break
            passo /= 2
        if passo <= 1e-3 or ll2 - ll < 1e-6:
            if passo > 1e-3:
                w = w2
            break
        w, ll = w2, ll2
    return w


def prever(w, cand, X):
    p = softmax([sum(a * b for a, b in zip(w, x)) for x in X])
    return dict(zip(cand, p))


# ---------------------------------------------------------------- calibração

def logit(c):
    c = min(max(c, 1e-6), 1 - 1e-6)
    return math.log(c / (1 - c))


def treinar_platt(pares, prior=0.5):
    """pares: lista de (confiança bruta, acertou 0/1). Ajusta P = σ(a·logit(c) + b).

    `prior` puxa (a, b) para (1, 0), isto é, para "confiar na probabilidade do modelo",
    o que evita uma curva extrema com poucos pontos.
    """
    a, b = 1.0, 0.0
    for _ in range(50):
        ga = -prior * (a - 1)
        gb = -prior * b
        haa = hbb = prior
        hab = 0.0
        for c, y in pares:
            z = logit(c)
            p = 1 / (1 + math.exp(-(a * z + b)))
            ga += (y - p) * z
            gb += (y - p)
            s = p * (1 - p)
            haa += s * z * z
            hab += s * z
            hbb += s
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a += da
        b += db
        if abs(da) + abs(db) < 1e-8:
            break
    return a, b


def calibrar(c, ab):
    a, b = ab
    return 1 / (1 + math.exp(-(a * logit(c) + b)))


# ---------------------------------------------------------------- confiança de publicar um nó

def conf_publicar(arv, probs, no):
    """P(publicar `no` é certo) segundo o modelo: massa dos rótulos para os quais `no` está no ramo."""
    return sum(probs.get(m, 0.0) for m in arv.ramo(no) if m != NULO)


def opcoes(arv, probs):
    """Devolve (mais específico, melhor tema nível 1) com as confianças brutas de publicação."""
    nos = arv.nos
    melhor = max((m for m in probs if m != NULO), key=lambda m: probs[m])
    temas = {}
    for m, p in probs.items():
        if m == NULO:
            continue
        t = m if nos[m]["nivel"] == 1 else nos[m]["pai"]
        if t in probs or t in nos:
            temas[t] = True
    melhor_tema = max(temas, key=lambda t: conf_publicar(arv, probs, t)) if temas else melhor
    return melhor, conf_publicar(arv, probs, melhor), melhor_tema, conf_publicar(arv, probs, melhor_tema)
