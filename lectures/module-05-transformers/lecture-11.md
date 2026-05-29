# Лекция 11: Архитектура Трансформер

## Модуль 5: LLM и Трансформеры

---

## 11.1 Проблема рекуррентных сетей

RNN (LSTM, GRU) обрабатывают последовательность шаг за шагом:

$$h_t = f(h_{t-1}, x_t)$$

### Фундаментальные ограничения

| Проблема | Причина |
|----------|---------|
| Длинные зависимости | Градиент затухает через 50-100 шагов |
| Последовательность | $O(n)$ — нельзя распараллелить по позициям |
| Узкое место | Всё состояние в одном векторе $h_t$ |

**Вывод**: для последовательности длины $n$ — $n$ последовательных операций. GPU простаивает.

### Attention как решение

Механизм attention (Bahdanau, 2014) показал: можно смотреть на **все** позиции сразу, взвешивая их по релевантности.

Трансформер (Vaswani et al., 2017): "Attention Is All You Need" — убираем рекурренцию полностью.

---

## 11.2 Self-Attention: базовая идея

### Интуиция

Каждое слово "смотрит" на все другие слова в предложении и решает, какие из них важны для его понимания.

> "Банк выдал кредит **банку**" — какое "банку"? Нужно посмотреть на контекст.

### Формула

Для последовательности $X \in \mathbb{R}^{n \times d}$ (n токенов, d — размерность):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Где:
- $Q = XW^Q$ — **queries** (что я ищу?)
- $K = XW^K$ — **keys** (что я предлагаю?)
- $V = XW^V$ — **values** (что я даю?)
- $W^Q, W^K \in \mathbb{R}^{d \times d_k}$, $W^V \in \mathbb{R}^{d \times d_v}$

### Пошаговое вычисление

Пусть $n=3$ токена, $d_k=4$:

**Шаг 1.** Вычисляем скоры:
$$s_{ij} = q_i \cdot k_j$$

Для каждого токена $i$ — скалярное произведение с каждым ключом $j$.

**Шаг 2.** Масштабируем:
$$s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

Зачем? При $d_k$ большом, скалярное произведение растёт, softmax насыщается → маленькие градиенты.

**Шаг 3.** Softmax по строкам:
$$\alpha_{ij} = \frac{e^{s_{ij}}}{\sum_j e^{s_{ij}}}$$

$\alpha_{ij}$ — вес, с которым токен $i$ "внимает" токену $j$.

**Шаг 4.** Взвешенная сумма:
$$\text{output}_i = \sum_j \alpha_{ij} v_j$$

### Размерности

$$Q, K \in \mathbb{R}^{n \times d_k}, \quad V \in \mathbb{R}^{n \times d_v}$$

$$QK^T \in \mathbb{R}^{n \times n} \quad \text{(матрица внимания)}$$

$$\text{softmax}(QK^T / \sqrt{d_k}) \cdot V \in \mathbb{R}^{n \times d_v}$$

### Численный пример

$X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$, $W^Q = W^K = W^V = I_2$

$$QK^T = XX^T = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}$$

Softmax (по строкам, $d_k=2$, $\sqrt{2} \approx 1.41$):

$$\text{scores} = \frac{1}{1.41}\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix} = \begin{bmatrix} 0.71 & 0 & 0.71 \\ 0 & 0.71 & 0.71 \\ 0.71 & 0.71 & 1.41 \end{bmatrix}$$

Токен 3 (строка 3) сильнее всего вниманияет самому себе — что логично, он содержит информацию обо всех.

### Живая визуализация: кто на кого смотрит

Возьмём предложение: **«Кот сидит на ковре»**

После применения self-attention получаем матрицу весов (уже после softmax):

```
         Кот   сидит   на    ковре
Кот     [0.55   0.25   0.05  0.15]
сидит   [0.20   0.40   0.15  0.25]
на      [0.05   0.15   0.30  0.50]
ковре   [0.10   0.20   0.40  0.30]
```

Визуализация (чем больше ▓, тем сильнее внимание):

```
         Кот    сидит    на     ковре
Кот     ▓▓▓▓▓  ▓▓▓     ░      ▓▓
сидит   ▓▓     ▓▓▓▓    ▓▓     ▓▓▓
на      ░      ▓▓      ▓▓▓    ▓▓▓▓▓
ковре   ▓      ▓▓      ▓▓▓▓   ▓▓▓
```

Что видим:
- «Кот» больше всего смотрит на себя (субъект) и на «сидит» (действие)
- «на» больше всего смотрит на «ковре» (предлог связывается с существительным)
- «ковре» смотрит на «на» (обратная связь) и на себя

Это один паттерн. Другая голова (head) могла бы ловить:
- синтаксис (подлежащее→сказуемое)
- близость (соседние слова)
- анафору (местоимение→антецедент)

Именно поэтому нужны multi-head — каждая голова «видит» свой тип связи.

---

## 11.3 Multi-Head Attention

### Зачем несколько голов?

Одна голова — один паттерн внимания. Но слова связаны по-разному:
- Синтаксис: подлежащее → сказуемое
- Семантика: "банк" → "кредит"
- Позиция: соседние слова

Multi-head позволяет изучить **несколько паттернов параллельно**.

### Формула

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Размерности

Пусть $d_{\text{model}} = 512$, $h = 8$ голов:

$$d_k = d_v = \frac{d_{\text{model}}}{h} = 64$$

Каждая голова работает в пространстве размерности 64 → дешевле, чем одна голова в 512.

### Параметры

| Компонент | Размерность | Параметров |
|-----------|-------------|------------|
| $W^Q$ | $512 \times 64 \times 8$ | 262,144 |
| $W^K$ | $512 \times 64 \times 8$ | 262,144 |
| $W^V$ | $512 \times 64 \times 8$ | 262,144 |
| $W^O$ | $512 \times 512$ | 262,144 |
| **Итого** | | **1,048,576 (1M)** |

### Реализация на NumPy

```python
import numpy as np

def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads):
    """
    X: (seq_len, d_model)
    Wq, Wk: (d_model, d_model) — объединённые для всех голов
    Wv: (d_model, d_model)
    Wo: (d_model, d_model)
    """
    seq_len, d_model = X.shape
    d_k = d_model // n_heads
    
    # Проекции
    Q = X @ Wq  # (seq_len, d_model)
    K = X @ Wk
    V = X @ Wv
    
    # Разбиваем на головы
    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)  # (n_heads, seq_len, d_k)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    
    # Scaled dot-product attention для каждой головы
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)  # (n_heads, seq_len, seq_len)
    attn = softmax(scores, axis=-1)
    heads = attn @ V  # (n_heads, seq_len, d_k)
    
    # Concat + линейная проекция
    heads = heads.transpose(1, 0, 2).reshape(seq_len, d_model)
    output = heads @ Wo
    
    return output, attn

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)
```

---

## 11.4 Positional Encoding

### Проблема

Self-attention **перестановочно**: переставь токены местами — получишь тот же результат (перемножение матриц не зависит от порядка строк).

> "Собака кусает человека" ≠ "Человек кусает собаку"

Нужно как-то дать модели информацию о позиции.

### Sinusoidal Encoding (оригинальный трансформер)

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

### Почему именно синус/косинус?

1. Каждая позиция получает **уникальный** вектор
2. Можно выразить относительные позиции: $PE(pos + k)$ — линейная функция от $PE(pos)$
3. Экстраполируется на длины, не_seen на обучении

### Визуализация паттерна

```
d_model=8, pos 0..50

dim 0 (sin, low freq):  ~~~~~~~~~~~~~~~~~~~~~~~~  (медленная волна)
dim 2 (sin, mid freq):  ~~~~~~~~~~~~~~~~~~~~~  (средняя)
dim 4 (sin, high freq): ~~~~~~~~~~~~~~  (быстрая)
dim 6 (sin, highest):   ~~ ~~ ~~ ~~ ~~  (очень быстрая)
```

Низкие частоты дают глобальную позицию, высокие — локальную.

### Learned Positional Embeddings

Альтернатива (используется в GPT, BERT):

$$E_{\text{pos}} \in \mathbb{R}^{n_{\text{max}} \times d_{\text{model}}}$$

Просто обучаемая матрица — по одной строке на каждую позицию.

| Метод | Плюсы | Минусы |
|-------|-------|--------|
| Sinusoidal | Нет параметров, экстраполяция | Фиксированный паттерн |
| Learned | Гибкость | Ограничена max_len |
| RoPE | Относительные позиции, экстраполяция | Сложнее в реализации |
| ALiBi | Линейный bias, простая экстраполяция | Менее распространён |

### Код

```python
def sinusoidal_pe(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len).reshape(-1, 1)
    div = 10000 ** (np.arange(0, d_model, 2) / d_model)
    
    pe[:, 0::2] = np.sin(pos / div)  # чётные
    pe[:, 1::2] = np.cos(pos / div)  # нечётные
    
    return pe

# Input = token embeddings + positional encoding
X = token_embeddings + sinusoidal_pe(seq_len, d_model)[:seq_len]
```

---

## 11.5 Полная архитектура Transformer

### Encoder Block

```
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm (residual + LayerNorm)     ← x + Sublayer(x)
  ↓
Feed-Forward Network (FFN)
  ↓
Add & Norm
  ↓
Output
```

### FFN

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

Обычно $d_{ff} = 4 \times d_{\text{model}}$:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times 4d}$
- $W_2 \in \mathbb{R}^{4d \times d_{\text{model}}}$

### Layer Normalization

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

В отличие от BatchNorm — нормализует по **фичам** одного примера, не зависит от батча.

### Residual Connection

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

Зачем: градиент течёт напрямую через $x$, минуя sublayer. Критично для глубины 24-96 слоёв.

### Decoder Block

Дополнительно:
1. **Masked Self-Attention** — нельзя смотреть вперёд
2. **Cross-Attention** — внимание на output encoder'а

Mask:
$$\text{scores}_{ij} = \begin{cases} s_{ij} & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

После softmax: $\alpha_{ij} = 0$ для $j > i$.

### Полная схема (Encoder-Decoder)

```
Source tokens → Embedding + PE → [Encoder Block × N] → Encoder output
                                                                ↓
Target tokens → Embedding + PE → [Decoder Block × N] → Linear → Softmax → Probabilities
                                         ↑
                              Cross-Attention (K,V from encoder)
```

### Подсчёт параметров (base model)

$d_{\text{model}} = 512$, $h = 8$, $d_{ff} = 2048$, $N = 6$ слоёв

| Компонент | Параметров |
|-----------|------------|
| Embedding | $V \times 512$ (V — словарь) |
| PE | 0 |
| 1 Encoder Block | |
| — Self-Attn (Q,K,V,O) | $4 \times 512^2 = 1{,}048{,}576$ |
| — FFN | $512 \times 2048 + 2048 \times 512 = 2{,}097{,}152$ |
| — LayerNorm × 2 | $2 \times 2 \times 512 = 2{,}048$ |
| — Итого 1 блок | ~3.1M |
| 6 Encoder blocks | ~18.7M |
| 6 Decoder blocks | ~25.0M (extra cross-attn) |
| Final linear | $512 \times V$ |
| **Итого (без embedding)** | **~44M** |

---

## 11.6 GPT: Decoder-Only

### Архитектура

GPT использует **только decoder** (без cross-attention):

```
Tokens → Embedding + PE
  ↓
[Masked Self-Attention + FFN] × N
  ↓
Linear (d_model → vocab_size)
  ↓
next token probability
```

### Зачем decoder-only?

Для генерации текста нужен только masked self-attention — предсказываем следующий токен, не глядя вперёд.

### Autoregressive генерация

На каждом шаге:
1. Берём последовательность $x_1, ..., x_t$
2. Прогоняем через трансформер
3. Берём логиты для позиции $t$
4. Семплируем $x_{t+1}$
5. Добавляем к входу, повторяем

```python
def generate(model, prompt_ids, max_new_tokens, temperature=1.0):
    ids = prompt_ids.copy()
    
    for _ in range(max_new_tokens):
        logits = model(ids)  # (1, seq_len, vocab_size)
        next_logits = logits[:, -1, :] / temperature
        probs = softmax(next_logits)
        next_id = np.random.choice(len(probs[0]), p=probs[0])
        ids.append(next_id)
    
    return ids
```

### Сравнение архитектур

| Модель | Тип | Задача | Пример |
|--------|-----|--------|--------|
| BERT | Encoder-only | Понимание текста | Классификация, NER |
| GPT | Decoder-only | Генерация | Текст, код |
| T5 | Encoder-Decoder | Seq2Seq | Перевод, саммаризация |
| LLaMA | Decoder-only | Генерация | Open-source LLM |

---

## 11.7 Ключевые математические свойства

### Сложность Self-Attention

$$O(n^2 \cdot d)$$

$n$ — длина последовательности. Матрица $QK^T$ имеет размер $n \times n$.

| Длина | Память (float32) |
|-------|------------------|
| 512 | 1 MB |
| 2,048 | 16 MB |
| 8,192 | 256 MB |
| 32,768 | 4 GB |
| 131,072 | 64 GB |

Поэтому контекст в 128K токенов — серьёзная инженерная задача.

### Эффективные аппроксимации

| Метод | Идея | Сложность |
|-------|------|-----------|
| Sparse Attention | Не все пары | $O(n\sqrt{n})$ |
| Linformer | Низкоранговая аппроксимация | $O(n \cdot k)$ |
| Flash Attention | Оптимизация I/O GPU | $O(n^2)$ но быстрее |
| Mamba/SSM | State space model | $O(n)$ |

### Permutation Equivariance

Self-attention **перестановочно** (без PE):

$$\text{Attn}(\pi(X)) = \pi(\text{Attn}(X))$$

Поэтому positional encoding обязателен.

### Эквивалентность с RNN

При $n \to \infty$ и определённой параметризации весов, линейный attention сходится к экспоненциальному ядру RNN. (Katharopoulos et al., 2020)

---

## 11.8 Практика

### 1. Self-Attention с нуля (NumPy)

```python
import numpy as np

def self_attention(X, d_k):
    """
    X: (seq_len, d_model)
    """
    seq_len, d_model = X.shape
    
    # Случайные веса (в реальности обучаются)
    np.random.seed(42)
    Wq = np.random.randn(d_model, d_k) * 0.1
    Wk = np.random.randn(d_model, d_k) * 0.1
    Wv = np.random.randn(d_model, d_k) * 0.1
    
    Q = X @ Wq  # (seq_len, d_k)
    K = X @ Wk
    V = X @ Wv
    
    # Scaled dot-product attention
    scores = (Q @ K.T) / np.sqrt(d_k)  # (seq_len, seq_len)
    
    # Softmax
    exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    
    output = attn_weights @ V  # (seq_len, d_k)
    
    return output, attn_weights

# Демонстрация
X = np.random.randn(5, 64)  # 5 токенов, d_model=64
output, attn = self_attention(X, d_k=32)

print("Attention weights shape:", attn.shape)  # (5, 5)
print("Output shape:", output.shape)           # (5, 32)
print("Сумма по строкам:", attn.sum(axis=1))    # все ~1.0
```

### 2. Transformer Block (PyTorch)

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, S, D = x.shape
        
        Q = self.Wq(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, D)
        
        return self.Wo(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        x = self.ln1(x + self.drop(self.attn(x, mask)))
        # FFN + residual + norm
        x = self.ln2(x + self.drop(self.ffn(x)))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, 
                 n_layers=4, d_ff=1024, max_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.tok_emb.weight = self.head.weight
        
        self.max_len = max_len
    
    def forward(self, idx):
        B, S = idx.shape
        
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(S, device=idx.device))
        x = tok + pos
        
        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=idx.device))
        mask = mask.view(1, 1, S, S)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

# Тест
model = GPT(vocab_size=1000, d_model=256, n_heads=8, n_layers=4)
total_params = sum(p.numel() for p in model.parameters())
print(f"Параметров: {total_params:,}")  # ~4.7M

x = torch.randint(0, 1000, (2, 64))  # batch=2, seq_len=64
logits = model(x)
print(f"Output: {logits.shape}")  # (2, 64, 1000)
```

### 3. Визуализация внимания

```python
import matplotlib.pyplot as plt

def visualize_attention(attn_matrix, tokens):
    """attn_matrix: (seq_len, seq_len)"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(attn_matrix, cmap='Blues', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    
    # Числа в ячейках
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f'{attn_matrix[i,j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    plt.colorbar(im)
    plt.title('Self-Attention Weights')
    plt.tight_layout()
    plt.savefig('attention_vis.png', dpi=150)
    plt.show()
```

---

## Резюме

| Концепция | Формула | Ключевое свойство |
|-----------|---------|-------------------|
| Scaled Dot-Product | $\text{softmax}(QK^T/\sqrt{d_k})V$ | $O(n^2)$ по памяти |
| Multi-Head | $\text{Concat}(\text{head}_1,..)W^O$ | Несколько паттернов |
| Positional Encoding | $\sin/\cos$ или learned | Порядок токенов |
| Residual + LN | $x + \text{Sublayer}(\text{LN}(x))$ | Стабильность градиента |
| Causal Mask | $-\infty$ для $j > i$ | Авторегрессия |

**GPT = Embedding + PE + N × (Masked Self-Attn + FFN) + Linear**

В следующей лекции — обучение LLM: кросс-энтропия, предобучение, RLHF, DPO.
