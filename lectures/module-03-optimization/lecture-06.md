# Лекция 6: Обратное распространение ошибки и регуляризация

## Модуль 3: Оптимизация и обучение моделей

---

## 6.1 Цепное правило дифференцирования

Backpropagation — это просто цепное правило, применённое к графу вычислений.

**Цепное правило для одной переменной:**

$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

**Пример:** $z = (3x + 1)^2$

Промежуточная переменная $y = 3x + 1$, тогда $z = y^2$.

$$
\frac{\partial z}{\partial y} = 2y = 2(3x + 1), \quad \frac{\partial y}{\partial x} = 3
$$

$$
\frac{\partial z}{\partial x} = 2(3x + 1) \cdot 3 = 6(3x + 1)
$$

**Цепное правило для нескольких переменных:**

$$
\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}
$$

---

## 6.2 Вычислительный граф

Любое выражение можно представить как ориентированный ациклический граф (DAG).

**Пример:** $f(x, y, z) = (x + y) \cdot z$

```
Прямой проход:

  x=2    y=-3    z=4
    \     /       |
     x+y=-1       |
       \         /
     (x+y)*z = -4    ← выход f

Обратный проход (градиенты):

  ∂f/∂(x+y) = z = 4
  ∂f/∂z = (x+y) = -1

  ∂f/∂x = ∂f/∂(x+y) · ∂(x+y)/∂x = 4 · 1 = 4
  ∂f/∂y = ∂f/∂(x+y) · ∂(x+y)/∂y = 4 · 1 = 4
```

**Ключевой принцип:** градиенты «текут» от выхода к входу, умножаясь на каждом узле.

---

## 6.3 Backpropagation для нейросети

### Постановка задачи

Двухслойная сеть:

$$
\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
$$

$$
\hat{\mathbf{y}} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2
$$

Функция потерь (MSE):

$$
L = \frac{1}{2}\|\hat{\mathbf{y}} - \mathbf{y}\|^2
$$

### Обратный проход по шагам

**Шаг 1.** Градиент по выходу:

$$
\frac{\partial L}{\partial \hat{\mathbf{y}}} = \hat{\mathbf{y}} - \mathbf{y}
$$

**Шаг 2.** Градиенты второго слоя:

$$
\frac{\partial L}{\partial \mathbf{W}_2} = \frac{\partial L}{\partial \hat{\mathbf{y}}} \cdot \mathbf{h}^T
$$

$$
\frac{\partial L}{\partial \mathbf{b}_2} = \frac{\partial L}{\partial \hat{\mathbf{y}}}
$$

$$
\frac{\partial L}{\partial \mathbf{h}} = \mathbf{W}_2^T \cdot \frac{\partial L}{\partial \hat{\mathbf{y}}}
$$

**Шаг 3.** Через активацию:

$$
\frac{\partial L}{\partial (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)} = \frac{\partial L}{\partial \mathbf{h}} \odot \sigma'(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
$$

где $\odot$ — поэлементное умножение.

**Шаг 4.** Градиенты первого слоя:

$$
\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)} \cdot \mathbf{x}^T
$$

$$
\frac{\partial L}{\partial \mathbf{b}_1} = \frac{\partial L}{\partial (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)}
$$

### Почему это эффективно

Наивный подход: считать каждый градиент отдельно $\Rightarrow O(n^3)$ на параметр.

Backprop: один прямой + один обратный проход $\Rightarrow O(n^2)$ суммарно.

| Метод | Сложность | Для сети 1M параметров |
|-------|-----------|----------------------|
| Численный градиент | $O(n^2)$ | ~1M прямых проходов |
| Backprop | $O(n)$ | 2 прохода |

---

## 6.4 Производные ключевых функций

Шпаргалка — понадобится при реализации backprop руками.

### Активации

**Sigmoid:** $\sigma(z) = \frac{1}{1+e^{-z}}$

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

**Tanh:** $\tanh(z)$

$$
\tanh'(z) = 1 - \tanh^2(z)
$$

**ReLU:** $\text{ReLU}(z) = \max(0, z)$

$$
\text{ReLU}'(z) = \begin{cases} 1, & z > 0 \\ 0, & z \leq 0 \end{cases}
$$

**GELU:** $\text{GELU}(z) = z \cdot \Phi(z)$, где $\Phi$ — CDF нормального распределения

$$
\text{GELU}'(z) = \Phi(z) + z \cdot \phi(z)
$$

где $\phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$ — плотность нормального распределения.

### Функции потерь

**MSE:** $L = \frac{1}{2}(\hat{y} - y)^2$

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y
$$

**Кросс-энтропия (бинарная):** $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$

$$
\frac{\partial L}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}
$$

**Softmax + кросс-энтропия (мультикласс):**

$$
\frac{\partial L}{\partial z_i} = p_i - y_i
$$

где $p_i = \text{softmax}(z_i)$, $y_i$ — one-hot. Это красивый результат — градиент упрощается до разности предсказания и истины.

---

## 6.5 Проблема исчезающих и взрывающихся градиентов

### Суть проблемы

При глубокой сети градиенты перемножаются на каждом слое. Если $|\sigma'| < 1$ — градиент затухает, если $> 1$ — растёт.

**Для sigmoid:** $\sigma'(z) \leq 0.25$ (максимум в нуле).

Сеть из 10 слоёв: $0.25^{10} \approx 10^{-6}$ — градиент исчезает.

**Для ReLU:** $\text{ReLU}'(z) \in \{0, 1\}$.

Нет затухания при $z > 0$, но «мёртвые нейроны» при $z \leq 0$.

### Решения

**1. Правильная инициализация весей:**

*Xavier (Glorot):* для $\tanh$

$$
W \sim \mathcal{U}\left[-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \quad \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}\right]
$$

*He (Kaiming):* для ReLU

$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)
$$

**2. Batch Normalization** (см. раздел 6.7)

**3. Residual connections** (ResNet — лекция 8)

**4. Gradient clipping:**

$$
\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \|\mathbf{g}\| \leq c \\ \frac{c}{\|\mathbf{g}\|}\mathbf{g} & \text{if } \|\mathbf{g}\| > c \end{cases}
$$

Типичное значение $c = 1.0$ или $c = 5.0$.

---

## 6.6 Регуляризация

### Зачем нужна

Цель модели — хорошо работать на новых данных (обобщение), а не только на обучающих.

**Переобучение (overfitting):** модель «запоминает» тренировочные данные.

$$
L_{train} \downarrow \downarrow, \quad L_{val} \uparrow
$$

### L2-регуляризация (Ridge, Weight Decay)

Добавляем штраф за большие веса:

$$
L_{reg} = L + \frac{\lambda}{2}\sum_{i} w_i^2 = L + \frac{\lambda}{2}\|\mathbf{w}\|_2^2
$$

Градиент:

$$
\frac{\partial L_{reg}}{\partial w_i} = \frac{\partial L}{\partial w_i} + \lambda w_i
$$

На каждом шаге вес «подтягивается» к нулю:

$$
w_i \leftarrow w_i - \eta(\nabla_i L + \lambda w_i) = (1 - \eta\lambda)w_i - \eta\nabla_i L
$$

Типичные значения $\lambda \in [10^{-5}, 10^{-2}]$.

### L1-регуляризация (Lasso)

$$
L_{reg} = L + \lambda\sum_{i}|w_i|
$$

Градиент (субградиент):

$$
\frac{\partial L_{reg}}{\partial w_i} = \frac{\partial L}{\partial w_i} + \lambda \cdot \text{sign}(w_i)
$$

**Ключевое отличие от L2:** L1 зануляет веса $\Rightarrow$ отбор признаков (sparsity).

| Свойство | L1 | L2 |
|----------|----|----|
| Решение | Разреженное | Плотное |
| Отбор признаков | Да | Нет |
| Устойчивость | Менее устойчив | Более устойчив |
| Аналитика | Нет closed-form | Есть closed-form |

### Elastic Net

Комбинация L1 и L2:

$$
L_{reg} = L + \lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|_2^2
$$

---

## 6.7 Dropout

### Идея

На каждом шаге обучения случайно «выключаем» нейроны с вероятностью $p$.

$$
\tilde{h}_i = \frac{r_i \cdot h_i}{1 - p}, \quad r_i \sim \text{Bernoulli}(1-p)
$$

Масштабирование на $1-p$ — чтобы математическое ожидание сохранялось: $\mathbb{E}[\tilde{h}_i] = h_i$.

### Почему работает

**Интуиция 1:** Каждый нейрон не может полагаться на конкретных соседей $\Rightarrow$ учится более устойчивым признакам.

**Интуиция 2:** Dropout ≈ ансамбль из $2^n$ подсетей, усреднённый на inference.

**Формально:** Dropout эквивалентен L2-регуляризации в пределе для линейных моделей (при определённых условиях).

### На практике

- Типичное $p = 0.5$ для скрытых слоёв, $p = 0.1\text{--}0.3$ для входного
- На inference dropout отключается, но масштабирование сохраняется
- В PyTorch: `nn.Dropout(p)` автоматически переключает режим

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),   # отключит 50% нейронов на train
    nn.Linear(256, 10)
)
```

---

## 6.8 Batch Normalization

### Проблема Internal Covariate Shift

Распределение входов каждого слоя меняется по мере обновления весов предыдущих слоёв. Это замедляет обучение — каждый слой адаптируется к «плывущему» распределению.

### Формула

Для мини-батча $\mathcal{B} = \{x_1, ..., x_m\}$:

$$
\mu_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^{m}x_i, \quad \sigma^2_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_{\mathcal{B}})^2
$$

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

где $\gamma$ и $\beta$ — обучаемые параметры (масштаб и сдвиг), $\epsilon = 10^{-5}$ — для численной стабильности.

### Что это даёт

| Свойство | Без BN | С BN |
|----------|--------|------|
| Скорость обучения | Медленнее | 2-10x быстрее |
| Чувствительность к LR | Высокая | Ниже |
| Исчезающие градиенты | Часто | Реже |
| Инициализация | Критична | Менее критична |

### На inference

ИспользуютсяRunning-статистики (экспоненциальное среднее за время обучения), а не батч-статистики.

---

## 6.9 Ранняя остановка и валидация

### Ранняя остановка (Early Stopping)

Мониторим $L_{val}$ на каждой эпохе. Останавливаемся, когда $L_{val}$ перестаёт падать.

```
Эпоха  | L_train | L_val  | Решение
-------|---------|--------|--------
  1    | 0.800   | 0.820  | продолжаем
  5    | 0.300   | 0.320  | продолжаем
  10   | 0.100   | 0.150  | продолжаем
  15   | 0.050   | 0.140  | ← patience
  20   | 0.020   | 0.160  | ← остановка!
```

**Patience** — количество эпох без улучшения до остановки. Типично 5-10.

### K-Fold кросс-валидация

Делим данные на $K$ частей (фолдов). Обучаем $K$ моделей, каждая на $K-1$ фолде, валидируемся на оставшемся.

$$
\text{CV score} = \frac{1}{K}\sum_{k=1}^{K} L_{val}^{(k)}
$$

Типично $K = 5$ или $K = 10$.

---

## 6.10 Практика

### Backprop руками на NumPy

```python
import numpy as np

# Данные (XOR)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Инициализация (He)
np.random.seed(42)
W1 = np.random.randn(2, 4) * np.sqrt(2/2)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * np.sqrt(2/4)
b2 = np.zeros((1, 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Обучение
lr = 0.5
for epoch in range(5000):
    # Прямой проход
    z1 = X @ W1 + b1
    h = sigmoid(z1)
    z2 = h @ W2 + b2
    pred = sigmoid(z2)
    
    # Потеря
    loss = np.mean((pred - y) ** 2)
    
    # Обратный проход
    d_pred = 2 * (pred - y) / len(y)           # dL/dpred
    d_z2 = d_pred * pred * (1 - pred)           # dL/dz2 (через sigmoid)
    
    d_W2 = h.T @ d_z2                           # dL/dW2
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)  # dL/db2
    
    d_h = d_z2 @ W2.T                           # dL/dh
    d_z1 = d_h * h * (1 - h)                    # dL/dz1 (через sigmoid)
    
    d_W1 = X.T @ d_z1                           # dL/dW1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)  # dL/db1
    
    # Обновление (SGD)
    W1 -= lr * d_W1
    b1 -= lr * d_b1
    W2 -= lr * d_W2
    b2 -= lr * d_b2

print("Предсказания:", pred.round(3).flatten())
# [0.048, 0.954, 0.954, 0.048]
```

### Сравнение регуляризаций на PyTorch

```python
import torch
import torch.nn as nn

# Модель с dropout и batchnorm
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10)
)

# L2 через weight_decay в оптимизаторе
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-3,
    weight_decay=1e-4    # это L2-регуляризация
)

# Early stopping
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(100):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        out = model(X_batch)
        loss = nn.CrossEntropyLoss()(out, y_batch)
        loss.backward()
        optimizer.step()
    
    # Валидация
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X_val, y_val in val_loader:
            out = model(X_val)
            val_loss += nn.CrossEntropyLoss()(out, y_val).item()
        val_loss /= len(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Сохраняем лучшую модель
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Ранняя остановка на эпохе {epoch}")
            break
```

---

## Итоги лекции

- **Backprop** = цепное правило + вычислительный граф. Сложность $O(n)$ вместо $O(n^2)$
- **Исчезающие градиенты** — главная проблема глубоких сетей. Решения: He-инициализация, BN, residual connections
- **L2** — штраф за большие веса, **L1** — отбор признаков через разреженность
- **Dropout** — случайное выключение нейронов, аналог ансамблевого усреднения
- **Batch Normalization** — нормализация по мини-батчу, ускоряет обучение в 2-10x
- **Early stopping** — простейшая и эффективная регуляризация
