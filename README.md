Nenhum selecionado


Pular para o conteúdo
Como usar o Gmail com leitores de tela
Ativar as notificações na área de trabalho para o Gmail.
   OK  Agora não(a)
Conversas
Gemini
Teste o Gemini no Gmail, Docs e outros apps
Turbine sua produtividade usando a IA e tenha acesso ao Google AI Pro e a 5 TB de armazenamento
3% de 400 GB usados
Termos · Privacidade · Regulamentos do programa
Última atividade da conta: há 0 minuto
Aberta em um outro local · Detalhes
# Explicação Didática do Projeto: Classificação de Sentimentos em Tweets sobre Saúde Mental

> Documento explicativo para estudantes iniciantes em programação.
> O objetivo aqui é traduzir o código original em **linguagem simples**, explicando **o que cada parte faz** e, principalmente, **por que ela foi feita assim**.

---

## 1. Visão Geral — O que esse projeto faz?

O projeto tenta resolver um problema clássico de **Machine Learning**: dado o texto de um tweet, descobrir se o autor está expressando sentimentos relacionados a **depressão (rótulo 1)** ou **não (rótulo 0)**.

Para isso, o autor experimentou **dois caminhos diferentes**:

1. **Caminho mais complexo (Deep Learning):** usar uma Rede Neural Convolucional (CNN) construída com TensorFlow/Keras.
2. **Caminho mais simples (ML clássico):** usar TF-IDF + Regressão Logística do scikit-learn.

> **Spoiler do final:** o caminho mais simples funcionou melhor. Isso é uma lição valiosíssima: **nem sempre a tecnologia mais "moderna" é a melhor escolha**. Voltaremos a esse ponto.

O fluxo geral do projeto, em qualquer um dos caminhos, segue sempre 4 etapas:

```
[Texto bruto] → [Limpeza] → [Vetorização] → [Modelo treina] → [Avaliação]
```

---

## 2. Bibliotecas Usadas — O kit de ferramentas

```python
import pandas as pd          # Manipulação de tabelas (planilhas)
import numpy as np           # Operações matemáticas com vetores
import tensorflow as tf      # Framework de Deep Learning (caminho complexo)
import re                    # Expressões Regulares (para limpar texto)
import matplotlib.pyplot as plt   # Gráficos básicos
import seaborn as sns        # Gráficos mais bonitos (em cima do matplotlib)
from sklearn... # Scikit-learn: a biblioteca de ML clássico
```

**Por quê essas bibliotecas?**

| Biblioteca               | Para quê serve                          | Alternativa mais simples                                    |
| ------------------------ | --------------------------------------- | ----------------------------------------------------------- |
| **pandas**               | Ler CSV, filtrar, agrupar dados         | `csv` (módulo padrão do Python) — mais verboso              |
| **numpy**                | Cálculos com arrays numéricos           | Listas do Python — bem mais lentas                          |
| **tensorflow**           | Construir e treinar redes neurais       | PyTorch (concorrente direto), ou só usar scikit-learn       |
| **re**                   | Encontrar e substituir padrões em texto | `str.replace()` — só funciona para texto exato, não padrões |
| **matplotlib / seaborn** | Visualização                            | `plotext` (gráficos no terminal), Plotly (mais moderno)     |
| **scikit-learn**         | Modelos de ML clássicos                 | É **A** biblioteca padrão. Não tem substituto à altura.     |

---

## 3. Carregamento dos Dados

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
CAMINHO_CSV = '/content/drive/My Drive/Mental-Health-Twitter.csv'
df = pd.read_csv(NOME_ARQUIVO).dropna(subset=[COLUNA_TEXTO, COLUNA_LABEL])
```

**O que está acontecendo aqui?**

1. `drive.mount(...)` — O notebook roda no **Google Colab** (uma máquina virtual gratuita do Google). Como essa máquina é "descartável" (some quando você fecha), os dados precisam vir de algum lugar persistente: o Google Drive.
2. `pd.read_csv(...)` — Lê o arquivo CSV e transforma numa "tabela" (chamada de **DataFrame**).
3. `.dropna(subset=[...])` — Remove linhas que tenham **valores faltando** (NaN = Not a Number) nas colunas que importam. Se um tweet está vazio ou o rótulo está faltando, ele é inútil para o treinamento.

**Por que se preocupar com nulos?**
Modelos de ML **quebram** quando recebem `NaN`. É como pedir para um aluno fazer uma prova com perguntas em branco — ele não tem como responder.

**Alternativa mais simples:** Em vez de Google Drive + Colab, você poderia rodar tudo localmente no seu PC com o arquivo na mesma pasta:
```python
df = pd.read_csv('Mental-Health-Twitter.csv').dropna(subset=['post_text', 'label'])
```
Pronto, tirou metade da complexidade.

---

## 4. Limpeza de Texto com Regex (a parte mais "mágica")

```python
def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'\@\w+', '', texto)
    texto = re.sub(r'\#', '', texto)
    texto = re.sub(r'[^a-záéíóúâêîôûãõç\s]', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()
```

**Regex (Regular Expressions)** = uma "linguagem de busca de padrões" dentro de textos. Parece complicado mas, traduzido linha por linha:

| Linha | O que faz | Por que faz |
|---|---|---|
| `.lower()` | Converte tudo para minúsculas | "Triste" e "triste" devem ser tratados como a mesma palavra |
| `re.sub(r'http\S+...', '', texto)` | Remove URLs | Links não carregam sentimento — são "ruído" |
| `re.sub(r'\@\w+', '', texto)` | Remove menções `@usuario` | Nomes de usuário não dizem nada sobre o sentimento do tweet |
| `re.sub(r'\#', '', texto)` | Remove o símbolo `#` (mas mantém a palavra) | `#triste` vira `triste`, que **é** uma palavra útil |
| `re.sub(r'[^a-záéí...]', ' ', texto)` | Remove tudo que não for letra | Pontuação, números, emojis viram espaço |
| `re.sub(r'\s+', ' ', ...)` | Normaliza espaços múltiplos em um só | Depois das limpezas anteriores, sobram espaços extras |

**Por que limpar tanto?**
Imagine que você está ensinando uma criança a reconhecer sentimentos lendo livros. Se você der pra ela uma página cheia de rabiscos, links e gibberish, ela vai se confundir. A limpeza tira o "barulho" e deixa só o sinal.

**Alternativa muito mais simples (sem regex):**
```python
def limpar_texto_simples(texto):
    texto = texto.lower()
    palavras = texto.split()
    palavras = [p for p in palavras if not p.startswith(('http', '@', '#'))]
    return ' '.join(palavras)
```
Funciona razoavelmente bem para casos simples e é **muito mais legível** para iniciantes. A regex só compensa quando você precisa de precisão cirúrgica.

**Alternativa profissional:** bibliotecas como **`spaCy`** ou **`NLTK`** fazem isso (e muito mais — lematização, stopwords, etc.) com uma linha:
```python
import nltk
texto_limpo = nltk.word_tokenize(texto.lower())
```

---

## 5. Divisão Treino/Teste

```python
X = df[COLUNA_TEXTO].astype(str).to_list()
y = df[COLUNA_LABEL].astype(int).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Conceito fundamental do ML:** Você **NUNCA** avalia um modelo nos mesmos dados que usou para treinar. É como dar a um aluno a prova com gabarito e elogiar ele por tirar 10.

- `X` = entradas (textos dos tweets)
- `y` = saídas esperadas (0 ou 1)
- `test_size=0.2` = 20% dos dados separados para teste, 80% para treino
- `random_state=42` = garante que toda vez que você rodar, a divisão seja **igual** (reprodutibilidade)

> Curiosidade: o número 42 não tem nada de especial. É uma piada de programador (referência ao "Guia do Mochileiro das Galáxias"). Poderia ser qualquer número.

---

## 6. Caminho A: Rede Neural Convolucional (CNN) — O Caminho Complexo

### 6.1. Tokenização e Padding

```python
vocab_size = 10000
max_length = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train),
                                maxlen=max_length, padding='pre')
```

**Problema:** Modelos não entendem texto. Entendem **números**.

**Solução:** Transformar cada palavra em um número.

- **Tokenizer**: Cria um "dicionário" — atribui um número único pra cada palavra (`"triste" → 47, "feliz" → 81`).
- `vocab_size = 10000`: Só mantém as 10.000 palavras mais frequentes. Palavras raras viram `<OOV>` (Out Of Vocabulary).
- **Padding**: Como os tweets têm tamanhos diferentes e o modelo exige tamanho **fixo**, todos viram listas de tamanho 100. Tweets curtos são preenchidos com zeros; tweets longos são cortados.
- `padding='pre'`: Os zeros vão no **começo** (não no fim). Isso é importante para redes recorrentes (LSTM) que tendem a "esquecer" o começo da sequência. Para CNN é menos crítico, mas é uma boa prática.

### 6.2. A Arquitetura da CNN

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32),
    tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001))
])
```

Vamos camada por camada, traduzindo de "matemática" para "intuição":

#### Camada 1: `Embedding` — "Significado das palavras"
Transforma cada número (que era uma palavra) em um **vetor de 32 dimensões** que representa o "significado" da palavra. Palavras parecidas terminam com vetores parecidos.

> **Analogia:** É como colocar palavras num mapa. "Triste" e "deprimido" ficam perto. "Feliz" fica longe de ambos.

#### Camada 2: `Conv1D` — "Detectores de padrões"
Aplica 64 "filtros" que deslizam pelo texto procurando **padrões de 7 palavras consecutivas** (`kernel_size=7`). Cada filtro aprende a reconhecer um padrão específico, como "não estou me sentindo bem" ou "estou muito feliz hoje".

> **Analogia:** É como ter 64 leitores especialistas, cada um treinado pra reconhecer um tipo específico de expressão.

#### Camada 3: `GlobalMaxPooling1D` — "Pega o pico"
Para cada um dos 64 filtros, pega só o **maior valor** detectado em todo o tweet. Isso responde: "esse padrão apareceu no tweet ou não?"

#### Camada 4: `Dropout(0.5)` — "Apagão proposital"
Durante o treino, **desliga aleatoriamente 50% dos neurônios** a cada passo. Parece loucura, mas funciona: força o modelo a não depender de neurônios específicos, generalizando melhor.

> **Analogia:** É como treinar um time de futebol fazendo cada jogador treinar de olhos vendados às vezes. Resultado: o time inteiro melhora.

#### Camada 5: `Dense(1, sigmoid)` — "Decisão final"
Um único neurônio com função `sigmoid` (que sempre devolve um número entre 0 e 1). Esse número é a **probabilidade** do tweet ser depressivo.

#### Regularização L2
`kernel_regularizer=l2(0.001)` — Pune o modelo se ele criar pesos muito grandes. Isso evita que ele "decore" os dados de treino (overfitting).

### 6.3. Compilação e Treinamento

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=5,
                                              restore_best_weights=True)

history = model.fit(X_train_padded, y_train, epochs=10, batch_size=64,
                    validation_split=0.2, callbacks=[early_stop])
```

- **Optimizer (Adam)**: O algoritmo que **ajusta os pesos** do modelo a cada erro. É como o "professor" que diz "tente ajustar mais pra esse lado".
- **Learning rate (0.0001)**: O tamanho do passo que o otimizador dá. Pequeno demais = aprende devagar. Grande demais = pula a resposta certa. Esse valor foi escolhido por ser bem conservador.
- **Loss `binary_crossentropy`**: A fórmula que mede o **erro** quando você tem duas classes (0 ou 1).
- **Epochs (10)**: Quantas vezes o modelo vê o dataset inteiro.
- **EarlyStopping**: Para o treino se o erro de validação não melhorar por 5 épocas. Evita perda de tempo e overfitting.

---

## 7. Caminho B: TF-IDF + Regressão Logística — O Caminho Simples (e Vencedor)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

log_reg_model = LogisticRegression(solver='liblinear', random_state=42, C=1.0)
log_reg_model.fit(X_train_tfidf, y_train)
```

### 7.1. O que é TF-IDF?

**TF-IDF** = **Term Frequency × Inverse Document Frequency** (Frequência do Termo × Frequência Inversa nos Documentos).

É uma forma de dar **peso** a cada palavra:

- **TF (Term Frequency):** Quantas vezes a palavra aparece no tweet. Aparece mais → mais importante.
- **IDF (Inverse Document Frequency):** Em quantos tweets a palavra aparece **em geral**. Se aparece em muitos → menos importante.

> **Intuição:** Palavras como "o", "de", "para" aparecem em todo lugar → peso baixo. Já palavras como "deprimido" ou "ansiedade" aparecem em poucos tweets específicos → peso alto e **discriminativo**.

```
TF-IDF = (vezes que palavra aparece no tweet) × log(total de tweets / tweets que têm a palavra)
```

**Resultado:** Cada tweet vira um vetor de 10.000 números, onde cada posição é o peso TF-IDF de uma palavra.

### 7.2. Regressão Logística

Apesar do nome assustador, é um dos modelos mais simples que existem. Em essência:

```
probabilidade = sigmoid(w1·palavra1 + w2·palavra2 + ... + wn·palavraN + b)
```

O modelo aprende **um peso `w` para cada palavra**. Palavras associadas a depressão ganham pesos positivos altos; palavras associadas a não-depressão ganham pesos negativos.

**Parâmetros importantes:**
- `solver='liblinear'`: Algoritmo eficiente para dados esparsos (TF-IDF gera muitos zeros).
- `C=1.0`: Força da regularização. Menor `C` = mais regularização = modelo mais simples.

### 7.3. Por que esse caminho ganhou?

A CNN tem **muito mais parâmetros** (pesos para aprender). Quando você tem **poucos dados** em comparação à complexidade do modelo, ele **decora** o treino em vez de **aprender** padrões generalizáveis. Isso é o **overfitting** — o modelo é um "gênio" no treino e um "burro" no teste.

A Regressão Logística é muito mais "humilde" — tem menos parâmetros, então só consegue aprender o que de fato generaliza.

> **Lição de ouro:** Comece sempre com o modelo mais simples possível. Só vá pra Deep Learning se for realmente necessário **e** você tiver muitos dados.

---

## 8. Avaliação dos Modelos

```python
y_pred = (model.predict(X_test_padded) > 0.5).astype(int).flatten()
print(classification_report(y_test, y_pred, target_names=nomes_classes))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

### 8.1. Matriz de Confusão

Uma tabela 2x2 que mostra **todos os tipos de erro e acerto**:

|  | Previu **Não Depressivo** | Previu **Depressivo** |
|---|---|---|
| **Real Não Depressivo** | Verdadeiro Negativo (TN) ✅ | Falso Positivo (FP) ❌ |
| **Real Depressivo** | Falso Negativo (FN) ❌❌ | Verdadeiro Positivo (TP) ✅ |

**No contexto de saúde mental, o erro mais grave é o Falso Negativo** — uma pessoa realmente em sofrimento ser classificada como "normal". Por isso o **recall** dessa classe é mais importante que a precisão geral.

### 8.2. Métricas do `classification_report`

| Métrica | Fórmula | Significado |
|---|---|---|
| **Precisão** | TP / (TP + FP) | Das vezes que disse "depressivo", quantas estavam certas? |
| **Recall** | TP / (TP + FN) | Dos depressivos reais, quantos o modelo pegou? |
| **F1-Score** | Média harmônica de precisão e recall | Equilíbrio entre os dois |
| **Acurácia** | (TP + TN) / Total | Proporção total de acertos |

---

## 9. Comparações com Tecnologias Mais Simples

Aqui está o cardápio "do mais simples ao mais complexo" para classificação de texto:

### 🟢 Nível 1: VADER ou TextBlob (sem treinar nada)
**Para quê?** Análise de sentimento básica usando dicionários pré-construídos. Você nem precisa de dataset!

```python
from textblob import TextBlob
texto = "I feel so sad and hopeless today"
polaridade = TextBlob(texto).sentiment.polarity  # -0.5 → negativo
```

**Prós:** Funciona em 2 linhas, sem treino.
**Contras:** Funciona mal para nuances específicas (depressão ≠ tristeza qualquer). Em inglês funciona bem; em português, pior.

### 🟢 Nível 2: Bag of Words + Naive Bayes
**Para quê?** O "Hello World" da classificação de texto.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vec = CountVectorizer(max_features=5000)
X_train_bow = vec.fit_transform(X_train)
modelo = MultinomialNB()
modelo.fit(X_train_bow, y_train)
```

**Prós:** Treina em milissegundos. Excelente baseline. Funciona bem em datasets pequenos.
**Contras:** Ignora ordem das palavras ("não estou triste" = "estou não triste" pra ele).

### 🟡 Nível 3: TF-IDF + Regressão Logística (o vencedor do projeto)
Já explicado acima. **É praticamente o melhor custo-benefício pra classificação de texto sem deep learning.**

### 🟡 Nível 4: TF-IDF + SVM (Support Vector Machine)
```python
from sklearn.svm import LinearSVC
modelo = LinearSVC(C=1.0)
modelo.fit(X_train_tfidf, y_train)
```
**Prós:** Frequentemente vence a Regressão Logística por uma pequena margem.
**Contras:** Mais lento em datasets grandes. Não dá probabilidades nativamente.

### 🟡 Nível 5: TF-IDF + Random Forest / Gradient Boosting (XGBoost, LightGBM)
**Prós:** Modelos muito potentes pra dados tabulares.
**Contras:** Para texto puro, raramente superam Regressão Logística com TF-IDF.

### 🔴 Nível 6: CNN / LSTM (o caminho que NÃO funcionou aqui)
Já explicado. Bom quando você tem **muitos** dados (>100k exemplos) **e** padrões sequenciais importantes.

### 🔴 Nível 7: BERT / Transformers (fine-tuning)
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis",
                       model="neuralmind/bert-base-portuguese-cased")
resultado = classifier("Estou me sentindo péssimo hoje")
```
**Prós:** Estado-da-arte. Entende contexto, ironia parcial, nuances.
**Contras:** Requer GPU, é lento, e para o problema deste projeto seria **matar mosca com canhão**.

### Tabela Resumo

| Abordagem | Linhas de código | Tempo de treino | Acurácia típica | Quando usar? |
|---|---|---|---|---|
| VADER/TextBlob | ~3 | 0s | 60-70% | Protótipo rápido |
| Bag of Words + Naive Bayes | ~10 | <1s | 70-80% | Baseline obrigatório |
| **TF-IDF + Regressão Logística** | **~15** | **<5s** | **75-85%** | **Quase sempre** ✅ |
| TF-IDF + SVM | ~15 | ~30s | 75-85% | Quando LR não basta |
| CNN/LSTM | ~50 | ~minutos | 70-90%* | Datasets grandes |
| BERT fine-tuned | ~30 | horas (GPU) | 85-95% | Produção crítica |

> *A CNN pode chegar a 90%+ se houver muitos dados, mas com poucos dados frequentemente fica abaixo de modelos mais simples por overfitting — exatamente o que aconteceu no projeto.

---

## 10. Lições para Levar para Casa

1. **Pré-processamento limpo > modelo complexo.** Os dados são mais importantes que o algoritmo.
2. **Sempre faça um baseline simples antes de partir pra deep learning.** Quantas vezes alguém percebeu que "não precisava de IA" só depois de gastar uma semana treinando uma rede neural?
3. **Train/test split não é opcional.** Avaliar no treino é se enganar.
4. **Overfitting é o inimigo invisível.** Um modelo com 99% no treino e 60% no teste é pior que um com 75%/75%.
5. **Reprodutibilidade importa.** Use `random_state` sempre.
6. **Métricas certas pro problema certo.** Em saúde mental, perder um caso real (Falso Negativo) é muito pior do que um falso alarme.

---

## 11. Sugestões Práticas para Estudar o Código

Se você está aprendendo agora, recomendo executar o projeto nessa ordem:

1. **Primeiro**, refaça só com TF-IDF + Logistic Regression (é menos de 30 linhas no total).
2. **Depois**, adicione a matriz de confusão e o classification_report.
3. **Depois**, troque a Regressão Logística por Naive Bayes e veja a diferença.
4. **Só então**, tente entender a parte de CNN — e veja com seus próprios olhos o overfitting acontecer.

Esse é o caminho natural de aprendizado: do simples ao complexo, observando **por que** o complexo nem sempre é melhor.

---

## Glossário Rápido

- **Tokenização:** quebrar texto em pedacinhos (geralmente palavras).
- **Embedding:** representação numérica densa de palavras (vetor).
- **Overfitting:** modelo decora o treino e falha no teste.
- **Regularização:** técnicas pra forçar o modelo a ser mais simples e evitar overfitting.
- **Hyperparâmetro:** configuração do modelo que **você** escolhe (não o modelo aprende). Ex: `C`, `learning_rate`, `kernel_size`.
- **Época (epoch):** uma passada completa pelo dataset durante o treino.
- **Batch:** subconjunto de amostras processado de uma vez.
- **Pipeline:** sequência de etapas (pré-processamento → vetorização → modelo → avaliação).
Explicacao_trabalho_depressao.md
Exibindo Explicacao_trabalho_depressao.md.
