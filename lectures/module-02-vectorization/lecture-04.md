# Лекция 4: Эмбеддинги и понижение размерности

## Содержание

1. [Что такое эмбеддинг](#1-что-такое-эмбеддинг)
2. [Word2Vec](#2-word2vec)
3. [GloVe](#3-glove)
4. [Понижение размерности: зачем](#4-понижение-размерности-зачем)
5. [PCA — метод главных компонент](#5-pca--метод-главных-компонент)
6. [t-SNE](#6-t-sne)
7. [UMAP](#7-umap)
8. [Универсальность эмбеддингов](#8-универсальность-эмбеддингов)
9. [Практика](#9-практика)

---

## 1. Что такое эмбеддинг

**Эмбеддинг** — отображение объекта (слово, картинка, пользователь) в вектор фиксированной размерности.

$$
\text{слово} \;\longrightarrow\; \mathbf{e} \in \mathbb{R}^d
$$

Зачем:
- Компактное представление (вместо one-hot на 50K слов — вектор на 300)
- Сохранение семантики: похожие объекты → близкие векторы
- Возможность арифметики: $\mathbf{e}_{\text{король}} - \mathbf{e}_{\text{мужчина}} + \mathbf{e}_{\text{женщина}} \approx \mathbf{e}_{\text{королева}}$

### One-hot vs эмбеддинг

One-hot для словаря $V = 50{,}000$:

$$
\mathbf{x}_{\text{кошка}} = [0, 0, 0, \ldots, 1, \ldots, 0] \in \mathbb{R}^{50000}
$$

Проблемы: разреженность, нет сходства, все векторы ортогональны.

### Bag of Words

Представим документ как мешок слов — считаем сколько раз каждое слово встретилось:

Словарь: [кошка, собака, мышь, бежит, спит]

Документ "кошка бежит кошка спит": $[2, 0, 0, 1, 1]$

Проблемы: потеря порядка, нет весов для редких/частых слов.

### TF-IDF

**TF** (term frequency) — как часто слово в документе:

$$
\text{TF}(t, d) = \frac{\text{число вхождений } t \text{ в } d}{\text{всего слов в } d}
$$

**IDF** (inverse document frequency) — насколько слово уникально across документов:

$$
\text{IDF}(t) = \log \frac{N}{|\{d \in D : t \in d\}|}
$$

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

Частые слова (the, и, а) — низкий IDF. Редкие и значимые — высокий.

Ограничения: всё ещё bag-of-words (нет порядка), размер вектора = размер словаря.

Эмбеддинг — следующий шаг: компактный + семантика + порядок через контекст.

$$
\mathbf{e}_{\text{кошка}} = [0.23, -0.41, 0.87, \ldots] \in \mathbb{R}^{300}
$$

Плотный, компактный,捕捉ает семантику.

### Матрица эмбеддингов

Для словаря $V$ и размерности $d$:

$$
E \in \mathbb{R}^{V \times d}
$$

Извлечение вектора слова с индексом $k$:

$$
\mathbf{e}_k = E[k, :] \in \mathbb{R}^d
$$

Или через one-hot: $\mathbf{e}_k = E^T \mathbf{x}_k$ — но на практике просто индексация.

---

## 2. Word2Vec

Два варианта: **CBOW** (контекст → слово) и **Skip-gram** (слово → контекст).

### Skip-gram

Идея: по слову предсказать его контекст.

$$
P(w_{\text{context}} \mid w_{\text{center}}) = \frac{\exp(\mathbf{u}_{\text{context}}^T \mathbf{v}_{\text{center}})}{\sum_{j=1}^{V} \exp(\mathbf{u}_j^T \mathbf{v}_{\text{center}})}
$$

Две матрицы:
- $E_{\text{in}} \in \mathbb{R}^{V \times d}$ — входные векторы (эмбеддинги)
- $E_{\text{out}} \in \mathbb{R}^{V \times d}$ — выходные векторы (контекст)

Функция потерь для окна размером $c$:

$$
\mathcal{L} = -\sum_{-c \le j \le c,\; j \ne 0} \log P(w_{t+j} \mid w_t)
$$

### Negative sampling

Полный softmax дорогой ($O(V)$). Negative sampling упрощает:

$$
\mathcal{L} = -\log \sigma(\mathbf{u}_{\text{pos}}^T \mathbf{v}) - \sum_{k=1}^{K} \log \sigma(-\mathbf{u}_{\text{neg}_k}^T \mathbf{v})
$$

где $K$ — число негативных примеров (5–20), $\sigma$ — сигмоида.

Выбор негативных примеров:

$$
P(w_i) = \frac{f(w_i)^{3/4}}{\sum_j f(w_j)^{3/4}}
$$

Степень $3/4$ сглаживает распределение — редкие слова получают шанс.

### CBOW

Обратная задача: по $2c$ словам контекста предсказать центральное слово.

$$
\hat{\mathbf{v}} = \frac{1}{2c}\sum_{j=-c}^{c} \mathbf{v}_{t+j}
$$

$$
P(w_t \mid \text{context}) = \text{softmax}(\mathbf{u}^T \hat{\mathbf{v}})
$$

CBOW быстрее (один forward на окно), Skip-gram лучше для редких слов.

---

## 3. GloVe

**Global Vectors** — объединяет локальный контекст (Word2Vec) и глобальную статистику.

### Матрица совместной встречаемости

$X_{ij}$ — сколько раз слово $j$ встретилось в контексте слова $i$.

### Функция потерь

$$
\mathcal{L} = \sum_{i=1}^{V}\sum_{j=1}^{V} f(X_{ij})\left(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2
$$

Весовая функция:

$$
f(x) = \begin{cases} (x / x_{\max})^{3/4} & \text{если } x < x_{\max} \\ 1 & \text{иначе} \end{cases}
$$

Зачем вес: без $f$ частые пары (the, of) доминируют, редкие — тонут.

### Итоговый эмбеддинг

$\mathbf{e}_i = \mathbf{w}_i + \tilde{\mathbf{w}}_i$ — сумма двух векторов даёт лучший результат.

---

## 4. Понижение размерности: зачем

Данные часто лежат в $\mathbb{R}^D$, но «истинная» размерность $d \ll D$.

Пример: лицо параметриуется 10–20 параметрами (наклон, освещение, выражение), но картинка — $10{,}000$ пикселей.

Зачем снижать размерность:
- **Визуализация** — человек видит в 2D/3D
- **Сжатие** — удаление шума
- **Ускорение** — меньше вычислений
- **Проклятие размерности** — при росте $D$ данные становятся разреженными

### Проклятие размерности

Объём $d$-мерного куба со стороной $\epsilon$:

$$
V = \epsilon^d
$$

При $\epsilon = 0.9$, $d = 100$: объём $0.9^{100} \approx 0.000027$ — почти все точки на границе.

Расстояние между случайными точками:

$$
\mathbb{E}[\|\mathbf{x} - \mathbf{y}\|_2] \approx \sqrt{d} \cdot \sigma
$$

При $d = 1000$ все расстояния примерно равны — теряется различимость.

---

## 5. PCA — метод главных компонент

### Интуиция

Найти направления максимальной дисперсии и спроецировать на них.

### Алгоритм

1. Центрирование: $\mathbf{X}_c = \mathbf{X} - \bar{\mathbf{X}}$
2. Ковариационная матрица: $C = \frac{1}{n-1}\mathbf{X}_c^T \mathbf{X}_c \in \mathbb{R}^{D \times D}$
3. Собственные векторы и значения: $C \mathbf{v}_k = \lambda_k \mathbf{v}_k$
4. Сортировка: $\lambda_1 \ge \lambda_2 \ge \ldots$
5. Проекция: $\mathbf{Z} = \mathbf{X}_c W$ где $W = [\mathbf{v}_1, \ldots, \mathbf{v}_k] \in \mathbb{R}^{D \times k}$

### Доля объяснённой дисперсии

$$
\text{explained ratio}_k = \frac{\lambda_k}{\sum_{j=1}^{D} \lambda_j}
$$

Суммарно для $k$ компонент:

$$
\text{cumulative}_k = \frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{D} \lambda_j}
$$

Практика: берём $k$ так, чтобы cumulative $\ge 0.95$ (95% дисперсии).

### SVD-подход

Через сингулярное разложение — быстрее и стабильнее:

$$
\mathbf{X}_c = U \Sigma V^T
$$

Главные компоненты: столбцы $V$. Проекция: $\mathbf{Z} = \mathbf{X}_c V_k$.

Связь: $\lambda_k = \sigma_k^2 / (n-1)$.

### Когда PCA работает хорошо

- Линейная структура данных
- Гауссово распределение
- Нужно быстрое сжатие

### Когда не работает

- Нелинейные структуры (swiss roll, спираль)
- Кластеры сложной формы

---

## 6. t-SNE

**t-Distributed Stochastic Neighbor Embedding** — нелинейное снижение до 2D/3D для визуализации.

### Шаг 1: Вероятности в пространстве признаков

$$
p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \ne i}\exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}
$$

Симметризация:

$$
p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}
$$

$\sigma_i$ подбирается так, чтобы perplexity $= 2^{-\sum_j p_{j|i}\log_2 p_{j|i}}$ был фиксирован (обычно 30).

### Шаг 2: Вероятности в пространстве embeddings

Student t-распределение с 1 степенью свободы (тяжёлые хвосты):

$$
q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \ne l}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}
$$

### Шаг 3: Минимизация KL-дивергенции

$$
\mathcal{L} = \text{KL}(P \| Q) = \sum_{i \ne j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

Градиент:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{y}_i} = 4\sum_j (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}
$$

### Почему t-распределение

Гаусс в high-D → гаусс в 2D: точки «схлопываются» в центр.

Тяжёлые хвосты t-распределия «расталкивают» удалённые точки → кластеры разделяются.

### Практические советы

- **Perplexity**: 5–50, по умолчанию 30
- **Не для feature engineering** — только визуализация
- **Разные запуски** дают разную форму кластеров
- **Размер кластера не имеет смысла** — только соседство
- **Сначала PCA до 50D**, потом t-SNE — ускорение и стабильность

---

## 7. UMAP

**Uniform Manifold Approximation and Projection** — быстрее t-SNE, сохраняет глобальную структуру.

### Идея

Предполагаем, что данные лежат на многообразии (manifold) с равномерной плотностью. Строим граф и оптимизируем layout.

### Шаг 1: Построение графа

Для каждой точки находим $k$ ближайших соседей.

Расстояние адаптируется:

$$
\tilde{d}_{ij} = d_{ij} - \rho_i
$$

где $\rho_i$ — расстояние до ближайшего соседа (гарантирует связность).

Вероятности (как в t-SNE, но с другими параметрами):

$$
p_{ij} = \exp\left(-\frac{\tilde{d}_{ij}}{\sigma_i}\right)
$$

Симметризация: $p_{ij} = p_{ij} + p_{ji} - p_{ij} \cdot p_{ji}$.

### Шаг 2: Оптимизация

Минимизируется кросс-энтропия:

$$
\mathcal{L} = -\sum_{(i,j)} \left[ p_{ij} \log q_{ij} + (1 - p_{ij})\log(1 - q_{ij}) \right]
$$

где $q_{ij}$ — через Student t-ядро (как в t-SNE).

### UMAP vs t-SNE

| Критерий | t-SNE | UMAP |
|----------|-------|------|
| Скорость | $O(n^2)$, медленный | $O(n \log n)$, быстрый |
| Глобальная структура | теряет | сохраняет лучше |
| Детерминизм | нет | почти (seed) |
| Размерность выхода | 2–3 | любая |
| Можно использовать для признаков | нет | да |
| Perplexity vs n_neighbors | 30 | 15 |

### Параметры UMAP

- **n_neighbors** (5–200) — баланс локальное/глобальное: меньше → локальная структура, больше → глобальная
- **min_dist** (0.0–0.99) — плотность точек: меньше → плотнее кластеры
- **metric** — евклид, косинус, и т.д.

---

## 8. Универсальность эмбеддингов

Эмбеддинги — не только про слова. Любой объект можно отобразить в векторное пространство.

### Изображения

CNN (например ResNet) на предпоследнем слое выдаёт вектор $\\mathbf{e} \\in \\mathbb{R}^{2048}$. Похожие картинки → близкие векторы. Так работают поиск по картинкам (Google Images) и face recognition.

### Аудио

Спектрограмма → CNN/Transformer → вектор. Голосовые ассистенты, Shazam, распознавание эмоций.

### Графы

Node2Vec, GraphSAGE: вершина графа → вектор. Социальные сети, рекомендательные системы.

### Модality-agnostic: CLIP

CLIP (OpenAI) — один Transformer для текста и картинок:

$$
\\text{sim}(\\text{image}, \\text{text}) = \\frac{\\mathbf{e}_{\\text{image}} \\cdot \\mathbf{e}_{\\text{text}}}{\\|\\mathbf{e}_{\\text{image}}\\| \\cdot \\|\\mathbf{e}_{\\text{text}}\\|}
$$

Общее пространство для текста и изображений → zero-shot классификация, поиск текст↔картинка.

### Общий паттерн

| Модальность | Модель | Размерность |
|-------------|--------|-------------|
| Текст | Word2Vec, BERT | 300–768 |
| Изображения | ResNet, ViT | 512–2048 |
| Аудио | Whisper, wav2vec | 512–1024 |
| Графы | Node2Vec, GNN | 128–256 |
| Мультимодальные | CLIP, ImageBind | 512–1024 |

Везде один принцип: объект $\\to$ вектор $\\to$ алгебра (сложение, расстояние, поиск ближайших).

---

## 9. Практика

### Word2Vec на маленьком корпусе (Gensim)

```python
from gensim.models import Word2Vec

sentences = [
    ["кот", "сидит", "на", "ковре"],
    ["собака", "лежит", "на", "ковре"],
    ["кот", "и", "собака", "друзья"],
    ["птица", "летит", "над", "домом"],
]

model = Word2Vec(sentences, vector_size=64, window=3, min_count=1, sg=1, epochs=100)

# Вектор слова
vec_cat = model.wv["кот"]  # shape: (64,)

# Похожие слова
similar = model.wv.most_similar("кот", topn=3)

# Аналогии: король - мужчина + женщина = ?
# model.wv.most_similar(positive=["король", "женщина"], negative=["мужчина"])
```

### PCA: сжатие + визуализация

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Данные: 1000 точек в 50D, но истинная размерность ~5
np.random.seed(42)
X = np.random.randn(1000, 5) @ np.random.randn(5, 50) + np.random.randn(1000, 50) * 0.1

# PCA
pca = PCA()
pca.fit(X)

# Доля объяснённой дисперсии
cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"5 компонент объясняют: {cumvar[4]:.2%}")
print(f"10 компонент объясняют: {cumvar[9]:.2%}")

# Проекция в 2D
X_2d = PCA(n_components=2).fit_transform(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.5)
plt.title("PCA: 50D → 2D")
plt.savefig("pca_2d.png", dpi=150)
```

### t-SNE vs UMAP на MNIST

```python
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from umap import UMAP

digits = load_digits()
X, y = digits.data, digits.target  # 1797 x 64

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# UMAP
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, X_emb, title in [(axes[0], X_tsne, "t-SNE"), (axes[1], X_umap, "UMAP")]:
    scatter = ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap="tab10", s=3, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

plt.colorbar(scatter, ax=axes, label="Цифра")
plt.savefig("tsne_vs_umap.png", dpi=150)
```

### Эмбеддинги: визуализация слов

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Упрощённые 3D-эмбеддинги (для иллюстрации)
words = {
    "кот":      [0.9, 0.1, 0.0],
    "собака":   [0.8, 0.2, 0.0],
    "животное": [0.5, 0.5, 0.0],
    "птица":    [0.3, 0.1, 0.8],
    "рыба":     [0.2, 0.0, 0.7],
    "машина":   [-0.5, 0.0, 0.0],
    "самолёт":  [-0.6, 0.0, 0.5],
}

X = np.array(list(words.values()))
labels = list(words.keys())

# PCA до 2D
X_2d = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1])
for i, label in enumerate(labels):
    plt.annotate(label, (X_2d[i, 0] + 0.02, X_2d[i, 1] + 0.02), fontsize=12)
plt.title("Семантическое пространство слов")
plt.grid(True, alpha=0.3)
plt.savefig("word_embeddings_2d.png", dpi=150)
```

---

## Ключевые формулы лекции

| Концепт | Формула |
|---------|---------|
| Skip-gram softmax | $P(w_c \mid w) = \frac{\exp(\mathbf{u}_c^T \mathbf{v})}{\sum_j \exp(\mathbf{u}_j^T \mathbf{v})}$ |
| Negative sampling | $\mathcal{L} = -\log\sigma(\mathbf{u}_{\text{pos}}^T\mathbf{v}) - \sum_k \log\sigma(-\mathbf{u}_{\text{neg}_k}^T\mathbf{v})$ |
| GloVe | $\sum_{ij} f(X_{ij})(\mathbf{w}_i^T\tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2$ |
| PCA проекция | $\mathbf{Z} = \mathbf{X}_c V_k$ |
| Доля дисперсии | $\frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{D}\lambda_j}$ |
| t-SNE KL | $\text{KL}(P \| Q) = \sum_{i \ne j} p_{ij}\log\frac{p_{ij}}{q_{ij}}$ |
| UMAP кросс-энтропия | $-\sum_{ij}[p_{ij}\log q_{ij} + (1-p_{ij})\log(1-q_{ij})]$ |
