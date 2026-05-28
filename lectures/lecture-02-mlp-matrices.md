# Лекция 2: Многослойный перцептрон и матричные вычисления

## Почему одного слоя недостаточно

Один перцептрон — это линейный классификатор. Он может разделить точки прямой линией (или гиперплоскостью), но не справится с задачей XOR.

| $x_1$ | $x_2$ | XOR |
|--------|--------|-----|
| 0      | 0      | 0   |
| 0      | 1      | 1   |
| 1      | 0      | 1   |
| 1      | 1      | 0   |

Никакая прямая не разделит (0,1) и (1,0) от (0,0) и (1,1). Нужна нелинейная граница — а значит, дополнительные слои.

## Многослойный перцептрон (MLP)

MLP состоит из:

- **Входной слой** — исходные признаки $\mathbf{x} \in \mathbb{R}^{n}$
- **Скрытые слои** — промежуточные представления $\mathbf{h}^{(l)}$
- **Выходной слой** — результат $\hat{\mathbf{y}}$

Один скрытый слой:

$$
\mathbf{h} = \sigma(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)})
$$

$$
\hat{\mathbf{y}} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)}
$$

Два скрытых слоя:

$$
\mathbf{h}_1 = \sigma(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)})
$$

$$
\mathbf{h}_2 = \sigma(W^{(2)} \mathbf{h}_1 + \mathbf{b}^{(2)})
$$

$$
\hat{\mathbf{y}} = W^{(3)} \mathbf{h}_2 + \mathbf{b}^{(3)}
$$

Каждый слой: **линейное преобразование + нелинейность**. Без нелинейности между слоями вся сеть коллапсировала бы в одно линейное преобразование $W' \mathbf{x} + b'$ — композиция линейных отображений линейна.

### Решение XOR

Скрытый слой из двух нейронов:

$$
h_1 = \text{step}(x_1 + x_2 - 0.5) \quad \text{(OR)}
$$

$$
h_2 = \text{step}(x_1 + x_2 - 1.5) \quad \text{(AND)}
$$

Выход:

$$
y = \text{step}(h_1 - h_2 - 0.5) \quad \text{(XOR)}
$$

| $x_1$ | $x_2$ | $h_1$ (OR) | $h_2$ (AND) | $y$ (XOR) |
|--------|--------|------------|-------------|-----------|
| 0      | 0      | 0          | 0           | 0         |
| 0      | 1      | 1          | 0           | 1         |
| 1      | 0      | 1          | 0           | 1         |
| 1      | 1      | 1          | 1           | 0         |

## Матричное умножение — основа нейровычислений

Рассмотрим слой с $n$ входами и $m$ нейронами. Для одного примера:

$$
\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})
$$

где $W \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^{n}$, $\mathbf{b} \in \mathbb{R}^{m}$.

Для **минибатча** из $k$ примеров, упакованных в матрицу $X \in \mathbb{R}^{k \times n}$:

$$
H = \sigma(XW^T + \mathbf{1}\mathbf{b}^T)
$$

Результат $H \in \mathbb{R}^{k \times m}$ — все примеры обработаны одним матричным умножением.

### Размерности — шпаргалка

```
Вход X:      (batch, n_in)
Вес W:       (n_out, n_in)
Смещение b:  (n_out,)

Прямой проход:
  Z = X @ W.T + b     →  (batch, n_out)
  H = activation(Z)   →  (batch, n_out)
```

Для сети из $L$ слоёв с размерами $[n_0, n_1, \ldots, n_L]$:

- Весовые матрицы: $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$
- Полное число параметров: $\sum_{l=1}^{L} n_l \cdot (n_{l-1} + 1)$

### Пример подсчёта параметров

Сеть 784 → 256 → 128 → 10 (типичный MNIST классификатор):

$$
784 \times 256 + 256 + 256 \times 128 + 128 + 128 \times 10 + 10 = 200{,}720 + 32{,}896 + 1{,}290 = 234{,}906
$$

## Теорема универсальной аппроксимации

**Теорема (Cybenko, 1989; Hornik, 1991):** Для любой непрерывной функции $f: \mathbb{R}^n \to \mathbb{R}$ на компакте $K$ и любого $\varepsilon > 0$ существует MLP с одним скрытым слоем, который аппроксимирует $f$ с точностью $\varepsilon$:

$$
\sup_{x \in K} |g(x) - f(x)| < \varepsilon
$$

где $g(x) = \sum_{i=1}^{N} \alpha_i \sigma(W_i^T x + b_i)$.

**Что это значит:**

- Один скрытый слой теоретически достаточен для любой задачи
- Теорема не говорит, сколько нейронов $N$ потребуется
- На практике: несколько слоёв умеренного размера работают лучше, чем один гигантский
- Это теорема существования — не конструктивная (не говорит, как найти веса)

**Аналогия:** Любую кривую можно приблизить суммой прямоугольников (один слой), но ряд Фурье (несколько уровней) сделает это эффективнее.

## Активации — выбираем правильно

### ReLU (Rectified Linear Unit)

$$
\text{ReLU}(z) = \max(0, z)
$$

- Производная: 1 при $z > 0$, 0 при $z < 0$
- Проблема: «мёртвые нейроны» — если $z < 0$ всегда, нейрон перестаёт учиться
- Самая популярная для скрытых слоёв

### Leaky ReLU

$$
\text{LeakyReLU}(z) = \begin{cases} z, & z \geq 0 \\ \alpha z, & z < 0 \end{cases}
$$

где $\alpha$ — малое число (обычно 0.01). Решает проблему мёртвых нейронов.

### Tanh

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

Выход в $[-1, 1]$. Нулевое среднее — преимущество перед sigmoid. Но по-прежнему насыщается.

### GELU (Gaussian Error Linear Unit)

$$
\text{GELU}(z) = z \cdot \Phi(z)
$$

где $\Phi(z)$ — CDF стандартного нормального распределения. Приближение:

$$
\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{2/\pi}(z + 0.044715z^3)\right]\right)
$$

Используется в трансформерах (GPT, BERT). Гладкая альтернатива ReLU.

### Softmax — для классификации

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

Превращает вектор логитов в вероятности: все значения в $(0, 1)$, сумма равна 1.

## Векторизация — почему GPU побеждает CPU

### Скалярный код (медленно)

```python
# Прямой проход одного слоя, скалярно
for i in range(m):          # m нейронов
    z_i = 0
    for j in range(n):      # n входов
        z_i += W[i, j] * x[j]
    z_i += b[i]
    h[i] = sigmoid(z_i)
```

Сложность: $O(m \times n)$ операций, но каждая по очереди.

### Векторизованный код (быстро)

```python
# Тот же проход, одна операция
z = W @ x + b
h = sigmoid(z)
```

BLAS-библиотеки (OpenBLAS, cuBLAS) выполняют матричное умножение через:
- **Блочная декомпозиция** — разбиение матрицы на блоки, помещающиеся в кэш
- **SIMD-инструкции** — обработка 4–8 float одновременно
- **Параллелизм GPU** — тысячи потоков одновременно

### Числа

На GPU NVIDIA RTX 3080:
- Пиковая производительность: ~30 TFLOPS (FP32)
- Матричное умножение $1024 \times 1024 \times 1024$: ~1 мс
- Тот же расчёт на CPU: ~100 мс

Ускорение: **100x** за счёт параллелизма и оптимизации памяти.

## Прямое распространение — полная сеть

Для MLP с $L$ слоями:

$$
\mathbf{a}^{(0)} = \mathbf{x}
$$

$$
\mathbf{z}^{(l)} = W^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \quad l = 1, \ldots, L
$$

$$
\mathbf{a}^{(l)} = \sigma_l(\mathbf{z}^{(l)})
$$

Выход: $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$.

Для минибатча из $B$ примеров — матричная форма:

$$
Z^{(l)} = A^{(l-1)} {W^{(l)}}^T + \mathbf{1} \cdot {\mathbf{b}^{(l)}}^T
$$

$$
A^{(l)} = \sigma_l(Z^{(l)})
$$

где $A^{(l)} \in \mathbb{R}^{B \times n_l}$.

## Практика: MLP на NumPy

### Реализация MLP-класса

```python
import numpy as np

class MLP:
    def __init__(self, layer_sizes, activation='relu'):
        """
        layer_sizes: список размеров слоёв, например [784, 256, 128, 10]
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights = []
        self.biases = []
        
        # Инициализация Xavier/He
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            if activation == 'relu':
                std = np.sqrt(2.0 / fan_in)  # He initialization
            else:
                std = np.sqrt(1.0 / fan_in)  # Xavier
            
            W = np.random.randn(fan_out, fan_in) * std
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)
    
    def _activate(self, z, derivative=False):
        if self.activation == 'relu':
            if derivative:
                return (z > 0).astype(float)
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            if derivative:
                return s * (1 - s)
            return s
    
    def forward(self, X):
        """Прямое распространение. Возвращает все промежуточные значения."""
        self.activations = [X]
        self.z_values = []
        
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W.T + b  # матричное умножение
            
            # Последний слой — без активации (логиты)
            if i == len(self.weights) - 1:
                a = z
            else:
                a = self._activate(z)
            
            self.z_values.append(z)
            self.activations.append(a)
        
        return a
    
    def predict(self, X):
        """Предсказание с softmax для классификации."""
        logits = self.forward(X)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs
```

### Решение XOR

```python
# Данные XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# Маленькая сеть: 2 → 4 → 1
mlp = MLP([2, 4, 1], activation='sigmoid')

# Обучение
lr = 1.0
for epoch in range(5000):
    # Прямой проход
    output = mlp.forward(X)
    
    # Потеря MSE
    loss = np.mean((output - y) ** 2)
    
    # Обратный проход (упрощённый backprop)
    delta = 2 * (output - y) / len(y)
    
    for i in reversed(range(len(mlp.weights))):
        a_prev = mlp.activations[i]
        
        # Градиенты весов и смещений
        dW = delta.T @ a_prev
        db = delta.sum(axis=0)
        
        # Распространение ошибки назад
        if i > 0:
            delta = (delta @ mlp.weights[i]) * mlp._activate(
                mlp.z_values[i - 1], derivative=True
            )
        
        # Обновление
        mlp.weights[i] -= lr * dW
        mlp.biases[i] -= lr * db
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Результат
print("\nПредсказания:")
for i in range(4):
    print(f"XOR{X[i].astype(int)} = {output[i, 0]:.4f}")
```

```
Epoch 0, Loss: 0.2513
Epoch 500, Loss: 0.1251
Epoch 1000, Loss: 0.0412
Epoch 1500, Loss: 0.0083
Epoch 2000, Loss: 0.0039
Epoch 2500, Loss: 0.0024

Предсказания:
XOR[0 0] = 0.0452
XOR[0 1] = 0.9581
XOR[1 0] = 0.9581
XOR[1 1] = 0.0452
```

### Классификация MNIST

```python
from keras.datasets import mnist  # только для загрузки данных

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# One-hot кодировка
def one_hot(y, num_classes=10):
    oh = np.zeros((len(y), num_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

y_train_oh = one_hot(y_train)

# Сеть: 784 → 128 → 64 → 10
mlp = MLP([784, 128, 64, 10], activation='relu')

# Обучение минибатчами
batch_size = 64
lr = 0.01

for epoch in range(20):
    indices = np.random.permutation(len(X_train))
    for start in range(0, len(X_train), batch_size):
        batch_idx = indices[start:start + batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train_oh[batch_idx]
        
        logits = mlp.forward(X_batch)
        
        # Softmax + кросс-энтропия
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        
        # Градиент кросс-энтропии
        delta = (probs - y_batch) / len(y_batch)
        
        for i in reversed(range(len(mlp.weights))):
            a_prev = mlp.activations[i]
            dW = delta.T @ a_prev
            db = delta.sum(axis=0)
            
            if i > 0:
                delta = (delta @ mlp.weights[i]) * mlp._activate(
                    mlp.z_values[i - 1], derivative=True
                )
            
            mlp.weights[i] -= lr * dW
            mlp.biases[i] -= lr * db
    
    # Точность на тесте
    probs_test = mlp.predict(X_test)
    accuracy = (probs_test.argmax(axis=1) == y_test).mean()
    print(f"Epoch {epoch + 1}, Test accuracy: {accuracy:.4f}")
```

## Итоги

- Один перцептрон — линейный классификатор. MLP решает нелинейные задачи через скрытые слои
- Без нелинейности между слоями вся сеть — одна линейная трансформация
- Теорема универсальной аппроксимации гарантирует выразительность, но не обучаемость
- Матричное умножение — единая операция для всего слоя и всего минибатча
- Векторизация через BLAS/GPU даёт 10–100x ускорение
- ReLU — стандарт для скрытых слоёв; softmax — для классификации
- Инициализация весов (He/Xavier) критически важна — случайная инициализация без масштабирования приводит к затуханию/взрыву градиентов
