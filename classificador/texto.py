"""Limpeza de texto, normalização, stemming leve e extração de citações legais.

Tudo determinístico e só com a biblioteca padrão.
"""
import html
import re
import unicodedata

STOPWORDS = set("""
a ao aos as ate com como da das de dei deu do dos e ela elas ele eles em entre era
essa essas esse esses esta estas este estes eu foi for ha isso isto ja la lhe lhes
mais mas me mesma mesmo meu minha muito na nas nao nem no nos nossa nosso num numa
o os ou para pela pelas pelo pelos por qual quando que quem se sem ser seu seus si
sob sobre sua suas tal tambem te tem ter um uma umas uns vos ja sera serao sido
pode podem deve devem caso cada todo toda todos todas outro outra outros outras
assinale alternativa alternativas correta corretas incorreta incorretas afirmar
afirmativa afirmativas assertiva assertivas item itens opcao certo errado julgue
seguinte seguintes acordo conforme respeito acerca relacao situacao hipotetica
considerando considere apenas nenhuma i ii iii iv v vi
""".split())

_ROMANOS = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


def limpar_html(s):
    if not s:
        return ""
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = s.replace("\xa0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", s).strip()


def sem_acento(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def normalizar(s):
    """Minúsculas, sem acentos, ordinais (º, °, ª) viram espaço."""
    s = sem_acento(s.lower())
    s = re.sub(r"[º°ª]", " ", s)
    return s


def radical(w):
    """Stemmer leve para português: une plural/gênero sem colidir termos distintos.

    furto/furtos -> furt; prisao/prisoes -> prisa; penal/penais -> penal;
    denuncia -> denunci, denunciacao -> denunciaca (não colidem).
    """
    if len(w) <= 3 or w.isdigit():
        return w
    for suf, rep in (("coes", "cao"), ("soes", "sao"), ("oes", "ao"), ("aes", "ao"),
                     ("ais", "al"), ("eis", "el"), ("ois", "ol"), ("ns", "m")):
        if w.endswith(suf):
            w = w[: -len(suf)] + rep
            break
    else:
        if w.endswith("s") and len(w) > 4:
            w = w[:-1]
    if len(w) > 4 and w[-1] in "aeo":
        w = w[:-1]
    return w


def tokens(s, manter_stop=False):
    """Texto bruto -> lista de radicais (sem stopwords por padrão)."""
    ws = re.findall(r"[a-z0-9]+", normalizar(s))
    out = []
    for w in ws:
        if not manter_stop and (w in STOPWORDS or len(w) == 1):
            continue
        out.append(radical(w))
    return out


def ngramas(toks, nmax=8):
    out = set()
    n = len(toks)
    for i in range(n):
        for k in range(1, nmax + 1):
            if i + k > n:
                break
            out.add(tuple(toks[i:i + k]))
    return out


# ---------------------------------------------------------------- citações legais

_DIPLOMAS = [
    (r"(?:do\s+)?(?:codigo\s+de\s+processo\s+penal|cpp)\b", "CPP"),
    (r"(?:do\s+)?(?:codigo\s+penal|cp)\b(?!\s*m)", "CP"),
    (r"(?:da\s+)?(?:lei\s+de\s+execucao\s+penal|lep)\b", "L7210"),
    (r"(?:da\s+)?(?:constituicao(?:\s+federal)?|cf(?:/88)?|crfb)\b", "CF"),
    (r"(?:do\s+)?(?:estatuto\s+da\s+crianca\s+e\s+do\s+adolescente|eca)\b", "L8069"),
    (r"(?:do\s+)?(?:codigo\s+de\s+transito(?:\s+brasileiro)?|ctb)\b", "L9503"),
    (r"(?:do\s+)?(?:codigo\s+de\s+defesa\s+do\s+consumidor|cdc)\b", "L8078"),
    (r"(?:do\s+)?codigo\s+penal\s+militar|cpm\b", "CPM"),
    (r"(?:do\s+)?codigo\s+de\s+processo\s+penal\s+militar|cppm\b", "CPPM"),
]

_RE_LEI = re.compile(
    r"\b(?:lei|decreto[\s-]*lei|dl|lc|lei\s+complementar)\s*(?:federal\s*)?(?:n\.?\s*|no\.?\s*|numero\s*)?"
    r"(\d{1,2})\.?(\d{3})(?:\s*(?:/|,?\s*de)\s*(\d{2,4}))?")
_RE_ART = re.compile(
    r"\bart(?:igo)?s?\.?\s*((?:\d{1,4}(?:\s*-\s*[a-z]\b)?(?:\s*,\s*|\s+e\s+|\s+a\s+)?)+)")
_RE_NUM = re.compile(r"(\d{1,4})(?:\s*-\s*([a-z])\b)?")


def _lei_id(m):
    return "L%s%s" % (m.group(1), m.group(2))


def extrair_leis(texto_norm):
    """Números de lei citados -> {'L11343', 'L9099', ...}."""
    return {_lei_id(m) for m in _RE_LEI.finditer(texto_norm)}


def chave_artigo(num, suf=""):
    return int(num) * 100 + ((ord(suf) - 96) if suf else 0)


def extrair_artigos(texto_norm):
    """Devolve lista de (diploma ou None, chave_artigo, explicito: bool).

    O diploma é procurado nos ~70 caracteres após a citação ("art. 155 do CP",
    "art. 33 da Lei 11.343/06"). Sem diploma explícito, devolve None e quem chama
    decide o diploma pelo contexto.
    """
    out = []
    for m in _RE_ART.finditer(texto_norm):
        nums = [(int(a), b or "") for a, b in _RE_NUM.findall(m.group(1))]
        if not nums:
            continue
        cauda = texto_norm[m.end(): m.end() + 70]
        dip = None
        ml = _RE_LEI.search(cauda)
        pos_lei = ml.start() if ml else 999
        for pat, cod in _DIPLOMAS:
            md = re.search(r"^\W*(?:,\s*)?(?:[^.;]{0,40}?)\b" + pat, cauda)
            if md and md.start() < pos_lei:
                dip = cod
                break
        if dip is None and ml and pos_lei < 45:
            dip = _lei_id(ml)
        for num, suf in nums:
            if num == 0 or num > 1000:
                continue
            out.append((dip, chave_artigo(num, suf), dip is not None))
    return out


_RE_FRASE_TEMA = re.compile(
    r"\b(?:no que (?:se refere|concerne|tange|diz respeito)|com (?:relacao|referencia)|em relacao"
    r"|a respeito|acerca|quanto|sobre|relativamente|no tocante|em materia)"
    r"\s+(?:(?:aos|ao|as|a|das|dos|da|do|de|os|o)\s+)?([^,.;:?!\n]{3,120})")


def frase_tema(enunciado, comando=""):
    """Assunto anunciado pela própria questão: "No que se refere ao Tribunal do Júri, ..." -> "tribunal do juri"."""
    for s in (enunciado[:400], comando or ""):
        m = _RE_FRASE_TEMA.search(normalizar(s))
        if m:
            return m.group(1)
    return ""


def diplomas_mencionados(texto_norm):
    return {cod for pat, cod in _DIPLOMAS if re.search(r"\b" + pat, texto_norm)}
