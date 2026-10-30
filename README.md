# Case Vício — classificação de tópicos

Sistema que decide, para cada questão, **publicar** num tema da árvore, mandar para **revisão** do
professor ou marcar como **fora** da taxonomia — sem chamar modelo de linguagem na decisão.

| Entrega | Arquivo |
|---|---|
| Decisões das 659 questões | [`publicacao.csv`](publicacao.csv) |
| Autoavaliação | [`previsao.txt`](previsao.txt) |
| Relatório (sistema, rótulos, diagnóstico, produção, memo ao professor) | [`relatorio.md`](relatorio.md) |
| Código | [`run.py`](run.py) + [`classificador/`](classificador/) |
| Conhecimento organizado (dado, editável) | [`kb/`](kb/) |
| Artefatos gerados (pesos, calibração, avaliação, diagnósticos) | [`artefatos/`](artefatos/) |

**Resultado (nas 559 questões fora da folha-verdade):** 217 publicadas (cobertura 0,39), 332 para
revisão, 10 fora. Precisão prevista 0,94 (intervalo 0,88–0,98); na validação cruzada aninhada sobre a
folha-verdade, a regra final acertou 96,4% das publicadas.

## Como rodar

Requer apenas **Python 3.10+** (biblioteca padrão; nenhuma dependência, nenhuma GPU, nenhuma rede).

```bash
python run.py
```

- Lê `anexos/` (os 5 arquivos do case, sem alteração) e `kb/`; escreve `publicacao.csv` e `artefatos/`.
- **Tempo:** ~25 s num notebook comum (≈2 s de features, ≈20 s de validação cruzada 10×5, <1 s para
  treinar o modelo final e decidir as 659 questões).
- `python run.py --rapido` pula a validação e usa a calibração salva em `artefatos/calibracao.json` (~3 s).
- Determinístico: duas execuções geram arquivos idênticos.
- `python run.py --anexos OUTRA_PASTA` roda sobre outra importação no mesmo formato.

## Como o sistema decide (resumo; detalhes no relatório)

1. **Evidências por par (questão, tema)** — `classificador/features.py`: termos do léxico do tema no
   texto, no assunto de origem e na "frase-tema" do enunciado; similaridade TF-IDF; artigos e leis
   citados (art. 155 do CP → Furto; Lei 11.343 → Tráfico); compatibilidade de disciplina; evidência
   propagada na árvore (pai/filhos); vizinhos rotulados da folha-verdade.
2. **Ranqueador** — `classificador/modelo.py`: softmax sobre os 351 nós candidatos (mais um "nenhum
   tema"), 23 pesos aprendidos com os 100 rótulos (central e extras valem como alvo).
3. **Confiabilidade do assunto de origem** (detector de lixeiras): para cada assunto, quanto ele
   concorda com o texto nas outras questões e com os rótulos da equipe. "Concurso de Crimes" = 0,14;
   "Lei Maria da Penha – Lei 11.340" = 0,86. Ver `artefatos/confiabilidade_assuntos.csv`.
4. **Calibração por grupo** — `classificador/decisao.py`: G1 = assunto confiável que aponta o mesmo
   ramo; G2 = sem assunto; G3 = o resto. G3 nunca é publicado (na folha, não passa de ~85% em nenhum
   limiar). Publica com confiança calibrada ≥ 0,80 (subtema exige ≥ 0,90; senão publica o tema pai).

## De onde veio cada artefato

| Artefato | Origem |
|---|---|
| `anexos/*` | Anexos do case, sem alteração. |
| `kb/lexico_temas.txt` | Termos e dispositivos legais de cada tema. **Rascunho gerado com LLM (Claude Opus 5.5, 2026-10-30) a partir só de `temas.csv` e do conhecimento dos códigos**, revisado e versionado como dado. Nenhuma questão foi usada para escrevê-lo. É o único uso de LLM, e é anterior à decisão. |
| `kb/nos_especiais.txt` | Decisão manual: 20 nós 67xx (vazios, sem contagem, título duplicado) nunca são destino; mapeados ao nó canônico. |
| `kb/convencoes.txt` | Decisão manual a partir da folha: questões de Juizado vão para Legislação Especial > JECrim (5/5 na folha). |
| `kb/disciplinas.txt` | Decisão manual: disciplina de origem → ramos compatíveis; Penal Militar e Processual Penal Militar → `fora` (sem ramo na árvore). |
| `artefatos/pesos_ranqueador.json` | Gerado por `run.py` (treino nos 100 rótulos). |
| `artefatos/calibracao.json` | Gerado por `run.py` (Platt por grupo nas previsões fora-da-dobra). |
| `artefatos/avaliacao_cv.json`, `folha_oof.csv` | Gerado por `run.py`: validação cruzada aninhada, tabela precisão × cobertura por limiar. |
| `artefatos/confiabilidade_assuntos.csv` | Gerado por `run.py`: o detector de lixeiras do site de origem. |
| `artefatos/decisoes_detalhadas.csv` | Gerado por `run.py`: caminho do tema, grupo, confiabilidade e motivo de cada decisão (para auditoria). |
| `artefatos/previsao_calculada.json` | Gerado por `run.py`: números usados no `previsao.txt`. |

## Decisões tomadas à mão (e por quê)

- **Nós duplicados/vazios** (`kb/nos_especiais.txt`): publicar num gêmeo vazio conta como erro quando a
  equipe usa o canônico. LEP em Processo Penal (6734) × Legislação Especial (4703): a equipe usa os
  dois; 4703 aparece 4 vezes contra 2 e foi o escolhido.
- **Convenção do Juizado** (`kb/convencoes.txt`): observada nos 5 casos da folha.
- **Militar → fora**: a árvore não tem Código Penal Militar nem Processo Penal Militar.
- **Limiar τ = 0,80**: meio do platô de precisão da validação aninhada (0,75–0,88 dão 96%; acima de
  0,90 a precisão cai, pois o erro de convenção restante tem confiança alta).
- **Nunca publicar G3**: escolha de regra, sustentada pela validação (ver relatório §a).

## Estrutura

```
run.py                     comando único
classificador/texto.py     limpeza de HTML, normalização, stemming, citações legais, frase-tema
classificador/kb.py        anexos, árvore, regra oficial de acerto, base de conhecimento
classificador/features.py  evidências por (questão, nó) e confiabilidade do assunto
classificador/modelo.py    ranqueador (softmax condicional, Newton/EM)
classificador/decisao.py   calibração por grupo e regra publicar / revisar / fora
classificador/pipeline.py  treino e validação cruzada
kb/                        conhecimento organizado (texto simples, editável pelo professor)
```
