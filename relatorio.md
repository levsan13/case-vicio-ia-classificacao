# Relatório — classificação de tópicos (case Vício, área de IA)

**Resultado.** Nas 559 questões fora da folha-verdade: 217 publicadas (cobertura 0,39 — 21% do banco,
54% da importação nova), 332 para revisão, 10 fora. Precisão prevista 0,94 (0,88–0,98). Na validação
cruzada aninhada sobre a folha-verdade, a regra final publica 28% e acerta 96,4%. Roda em ~25 s, só com
a biblioteca padrão do Python, sem modelo de linguagem na decisão.

## a) O sistema

A ideia central: **uma questão só é publicada quando duas fontes independentes concordam** — o texto
da questão (lido pelo sistema) e o assunto que o site de origem atribuiu — e só se esse assunto for
historicamente confiável. Todo o resto vai para o professor com uma sugestão e um motivo.

**1. Conhecimento organizado (`kb/`, texto simples).** Para cada um dos 375 temas ativos: termos
característicos e dispositivos legais (CP, CPP, leis especiais — p.ex. Furto = CP 155–156; Juiz das
Garantias = CPP 3-A a 3-F; Tráfico = Lei 11.343). O rascunho foi gerado por LLM a partir **só** da árvore
e dos códigos, antes de qualquer decisão, e revisado; é o único uso de LLM. Também em `kb/`: os 20 nós
duplicados/vazios que nunca são destino, a convenção do Juizado e o mapa de disciplinas.
*Por quê:* as descrições de `temas.csv` são genéricas ("Subtema de X. N questões no source"); sem
enriquecer os temas, não há o que comparar com o texto.

**2. Evidências por par (questão, tema)** — 23 features interpretáveis: termos do léxico encontrados
no texto (pesados pela especificidade do termo na árvore); idem no assunto de origem e na "frase-tema"
que a própria questão anuncia ("No que se refere ao Tribunal do Júri…"); similaridade TF-IDF; artigo
citado com diploma explícito ("art. 155 do CP") ou inferido pelo contexto; número de lei citado;
compatibilidade de disciplina; a mesma evidência propagada na árvore (o tema herda do melhor subtema);
vizinhos rotulados (questão parecida ou mesmo assunto na folha-verdade).

**3. Ranqueador.** Softmax condicional sobre os 351 nós candidatos mais um candidato "nenhum tema",
com 23 pesos aprendidos nos 100 rótulos (central e extras valem como alvo; EM + Newton, L2). Poucos
parâmetros para poucos rótulos, e cada peso é legível (`artefatos/pesos_ranqueador.json`: maiores
pesos em similaridade com o tema, lei citada, vizinhos rotulados, artigo explícito).

**4. Detector de lixeiras do site de origem.** Para cada `assunto_origem`, mede-se em quantas *outras*
questões com o mesmo assunto ele concorda com o que o texto indica (sem rótulo, nas 659) e com o
rótulo da equipe (questões da folha, peso 3), suavizado pela média da classe de frequência (assuntos
únicos concordam ~75%; os que aparecem 10+ vezes no banco, ~30%). Resultado em
`artefatos/confiabilidade_assuntos.csv`: "Concurso de Crimes" (44 questões) r = 0,14; "Ação Penal
(Parte 02)" 0,11; "Legislação Penal Especial" 0,03; "Lei Maria da Penha – Lei 11.340" 0,86.

**5. Calibração e decisão.** Três grupos: **G1** assunto confiável (r ≥ 0,7) que aponta o mesmo ramo
escolhido; **G2** sem assunto; **G3** o resto. Confiança = Platt por grupo sobre a probabilidade do
ranqueador, ajustado nas previsões fora-da-dobra. Publica-se G1/G2 com confiança ≥ 0,80, no subtema se
ele tiver ≥ 0,90, senão no tema pai (que pela regra também conta como certo). G3 sempre vai para revisão.
Militar → `fora` (sem ramo na árvore).

**Validação (honesta).** Validação cruzada 10 dobras × 5 repetições, **aninhada**: ranqueador e
calibração são ajustados sem a dobra avaliada. Acerto do 1º lugar (com crédito de ramo): 71%.

| limiar τ | 0,60 | 0,70 | 0,75 | **0,80** | 0,85 | 0,90 | 0,94 |
|---|---|---|---|---|---|---|---|
| cobertura (folha) | 0,30 | 0,29 | 0,29 | **0,28** | 0,27 | 0,17 | 0,05 |
| precisão (folha) | 0,93 | 0,94 | 0,97 | **0,96** | 0,96 | 0,94 | 0,80 |

**Testado e descartado.**
- *Publicar por confiança do texto, sem olhar o assunto:* a precisão estaciona em ~86% em qualquer
  limiar — os erros mais confiantes são rótulos "fora" e convenções, não confusão do modelo.
- *Marcar como lixeira o assunto frequente (≥ 6 questões):* errado no lote novo, onde "Lei Maria da
  Penha – Lei 11.340" é frequente e específico. Substituído pela confiabilidade medida (item 4).
- *Calibração logística única com termos aditivos:* não consegue expressar "em G3 a confiança do texto
  não ajuda"; colocava os erros de G3 no topo. Substituída pelo Platt por grupo.
- *Publicar G3 com limiar alto:* nenhum limiar passa de ~85% (os casos "certeiros" para o texto em G3
  são justamente os que a equipe rotulou de outro jeito).
- *Limiar acima de 0,90:* só reduz cobertura (o erro restante, Q0202, tem confiança alta).
- *Feature "parte geral do CP"* (para a convenção conceito × crime): mantida, mas o peso aprendido é
  ~0 — um exemplo só não ensina.
- *Bug encontrado:* "(Parte 02)" no assunto casava com o tema "Sujeitos do Processo" ("partes"); hoje a
  paginação do site é removida antes da comparação.
- *Não testado:* embeddings densos (exigiriam baixar um modelo; o léxico + TF-IDF já separa bem os temas
  e a limitação principal está nos rótulos, não na representação). É o primeiro teste com mais tempo.

## b) Os 100 rótulos

**Como foram usados.** (1) Treino dos 23 pesos do ranqueador; (2) memória de vizinhos (questões
parecidas e mesmo assunto); (3) parte supervisionada da confiabilidade do assunto; (4) calibração por
grupo; (5) validação aninhada, que gera a tabela acima e a previsão; (6) extração de convenções
(Juizado → Legislação Especial; LEP → 4703). Toda métrica reportada é fora-da-dobra.

**O que mostraram sobre o sistema.** O ranqueador acerta o ramo em ~71% das vezes, mas a confiança do
texto sozinha não separa acertos de erros no topo. O que separa é a concordância com um assunto de
origem confiável: em G1 (26 das 100), 96% de acerto; em G3 (69), ~65% e plano.

**O que mostraram sobre a convenção da equipe.**
- **"Fora" é inconsistente.** 8 rótulos "fora"; só 3 têm motivo claro na árvore (ordem econômica, com
  "Direito penal econômico" inativo; procedimento dos crimes de responsabilidade de funcionário
  público, sem nó; um de criminologia). Os outros 5 parecem classificáveis: Q0303 "aplicação da lei
  penal" (há Lei Penal no Tempo e no Espaço), Q0043 "fixação da pena", Q0087 prisão domiciliar na
  execução, Q0252, Q0291. **Todos os 8 vêm de assuntos-lixeira do site** (Ação Penal…, Teoria da Pena
  (Parte 02), Concurso de Crimes) — por isso o sistema não publica G3.
- **Conceito testado > crime do enunciado.** Q0202 narra favorecimento da prostituição, mas testa erro de
  tipo, e é rotulada como Erro de Tipo. É o único erro da regra final na validação.
- **Nós duplicados são usados dos dois lados.** LEP aparece em Legislação Especial (4×) e em Processo
  Penal (2×); "Teoria da Pena (Parte 02)" vira LEP em 4 de 7 casos.
- **Juizado:** 5/5 em Legislação Especial > JECrim, com nós de Processo Penal só como extras.
- **A equipe usa 93 dos 373 nós.** Os 8 maiores subtemas da plataforma (Conceito e Fontes do Direito
  Penal, 310 questões; Sistemas Processuais Penais, 285; Princípios do Processo Penal, 253; Lei penal no
  Tempo, 227; Características e Condições da Ação Penal, 212; Conceitos e Espécies de Prisão, 209…)
  aparecem **zero** vezes nos 100 rótulos, quando se esperariam ~19 se refletissem conteúdo real. São
  as lixeiras de que o case fala.
- Os 100 rótulos são todos do banco (Q0001–Q0359); nenhum da importação nova.

## c) Diagnóstico

**O que atrapalha, nos dados:** assuntos de origem que são lixeiras (no banco, 53% das questões têm
assunto com r < 0,5: 191 de 359); paginação do site no assunto ("Parte 01/02"); HTML cru com estilos; 19 questões
sem assunto; disciplina de origem errada (questões de "Direito Civil – Legislação Especial" que são de
inquérito policial).

**O que atrapalha, na árvore:** 20 nós 67xx vazios que duplicam outros ("Teoria da Pena (Parte 1/2)",
"Competência (Parte 1/2)", "Provas em Espécie" × "Provas em Espécies"); temas duplicados entre
disciplinas (LEP, Ação Penal, Lavagem, Maria da Penha, Tráfico, Organizações Criminosas, Estatuto do
Idoso, Abuso de Autoridade, Meio Ambiente); Prescrição como irmã de Punibilidade (é causa de extinção
da punibilidade); subtema "Fato Típico" com o mesmo nome do pai; nenhum nó para procedimentos especiais
(funcionário público, crimes contra a honra), penal econômico (inativo) e Direito Penal Militar; títulos
com erro ("Crimonológicas", "Comeptência", "Estaturo de Idoso", "Caractéristas", "Crimescontra").

**O que o sistema ainda erra, e o que resolveria:**

| Tipo de erro | Exemplo | O que resolveria |
|---|---|---|
| Rótulo "fora" em questão classificável | Q0303, Q0043 | Critério escrito para "fora" (e para "questão desatualizada", se for isso) |
| Conceito testado × cenário | Q0202 | Regra da equipe por escrito + 20 exemplos rotulados desse tipo |
| Nós duplicados | LEP 4703 × 6734 | Fundir ou desativar os 67xx; um único nó por lei |
| Questões multitema | Q0221, Q0325 | Usar mais os extras; ou tema "Revisão geral" explícito |
| Assunto-lixeira sem texto decisivo | 256 revisões | Rótulos do professor nessas filas (cada um melhora o r do assunto inteiro) |
| Lote novo sem rótulo | 300 questões | 30–50 rótulos da importação nova para medir/ajustar |

## d) Produção

**Custo e tempo para 50 mil questões/semana.** As features custam ~3 ms por questão num núcleo (351
nós × termos); 50 mil ≈ 2,5 min de CPU, sem GPU, sem API, sem rede — custo marginal ~zero. O treino
leva <1 s e a validação completa ~20 s. Cabe numa tarefa agendada simples.

**Correção do professor → dado de treino.** Cada questão revisada vira uma linha no mesmo formato de
`folha_verdade.csv` (central, extras, fora). Ela alimenta, sem código novo: os pesos do ranqueador, a
memória de vizinhos, a confiabilidade do assunto (uma correção num assunto-lixeira corrige o r de todas
as questões daquele assunto) e a calibração. Correções recorrentes viram dado em `kb/` (termo novo no
léxico, convenção nova). Priorização da fila: primeiro as questões de assuntos com muitas questões e r
incerto — cada rótulo destrava um assunto inteiro.

**Retreino.** Semanal, junto com a importação: retreina, roda a validação aninhada e **só troca o
modelo se a precisão fora-da-dobra nos últimos rótulos ficar ≥ 95%**; o limiar τ é recalibrado no mesmo
passo. Monitoramento: auditoria semanal de ~30 publicadas sorteadas, estratificadas por lote/fonte; se
a precisão auditada cair abaixo de 95%, sobe-se τ até a próxima rodada. Revisão do léxico e da árvore
com o professor a cada trimestre.

---

## e) Memo para o professor

**Para:** professor responsável pelas questões de Penal · **Assunto:** classificação automática

**O que fizemos.** Montamos um sistema que lê cada questão e propõe o tema. Ele só publica direto
para o aluno quando duas coisas concordam: o que a questão diz (termos, artigos e leis citados) e o
assunto que veio do site de origem — e só se esse assunto costuma ser confiável. Nesta rodada, das 559
questões, publicamos 217 (39%), mandamos 332 para o senhor com uma sugestão de tema e um motivo, e 10
ficaram de fora (Direito Penal Militar, que não existe na árvore). Esperamos que ~94% das publicadas
estejam certas; na amostra de 100 que a equipe classificou, a regra acertou 96%.

**O que encontramos.**
1. **As lixeiras existem e são mensuráveis.** Os oito maiores subtemas da plataforma (como "Conceito e
   Fontes do Direito Penal", com 310 questões) não aparecem nenhuma vez entre as 100 questões que a
   equipe classificou. Também no site de origem há "assuntos-lixeira": "Concurso de Crimes" e
   "Ação Penal (Parte 01/02)" quase nunca correspondem ao conteúdo da questão.
2. **A árvore tem temas repetidos.** "Teoria da Pena (Parte 1)" e "(Parte 2)", "Competência (Parte 1/2)",
   "Provas em Espécie" e "Provas em Espécies" estão vazios e duplicam outros. A LEP existe em Processo
   Penal e em Legislação Especial, e a equipe usou os dois.
3. **"Fora da taxonomia" foi usado de formas diferentes.** Algumas questões marcadas "fora" parecem
   caber num tema existente (ex.: "aplicação da lei penal").

**O que recomendamos mudar nos temas.**
- Apagar ou fundir os 20 temas vazios duplicados e manter um único tema por lei (LEP, Ação Penal,
  Lavagem, Maria da Penha, Tráfico…).
- Redistribuir as questões dos oito subtemas-lixeira (começar pelos 310 de "Conceito e Fontes").
- Criar temas para o que hoje não tem lugar: procedimentos especiais (funcionário público, crimes contra
  a honra), crimes contra a ordem econômica (o tema "Direito penal econômico" está inativo) e, se fizer
  parte do produto, Direito Penal Militar.
- Corrigir títulos com erro de digitação ("Crimonológicas", "Comeptência", "Estaturo de Idoso").

**Do que precisamos do senhor.**
1. Uma regra escrita para dois casos: quando a questão narra um crime mas testa um conceito da parte
   geral (erro de tipo, tentativa), qual tema vale? E o que é "fora da taxonomia"?
2. Classificar 30 a 50 questões da importação nova — hoje não temos nenhum exemplo dela, e é de lá que
   vem a maior parte do que publicamos.
3. Responder a fila de revisão no formato que já existe: cada resposta ensina o sistema e, quando é de
   um assunto repetido, corrige de uma vez todas as questões daquele assunto.
4. Uma revisão do vocabulário de cada tema (`kb/lexico_temas.txt`, texto simples): termos faltando
   ou errados ali são a causa mais barata de corrigir.
