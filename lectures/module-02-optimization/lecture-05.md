# Лекция 5: Градиентный спуск

> Модуль 2. Принципы обучения моделей

---

## 1. Что такое градиент

Градиент — вектор частных производных функции по всем переменным:

$$
\nabla f(\mathbf{w}) = \left(\frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \ldots, \frac{\partial f}{\partial w_n}\right)
$$

Градиент указывает направление **наискорейшего роста** функции. Антиградиент $-\nabla f$ — направление наискорейшего убывания.

### Геометрическая интуиция

Представим функцию потерь как горный ландшафт. Градиент в каждой точке — стрелка, показывающая куда подниматься. Мы идём в обратную сторону — вниз.

### Для функции одной переменной

$$
f(w) = w^2 - 4w + 3
$$

Производная:

$$
f'(w) = 2w - 4
$$

В точке $w = 0$: $f'(0) = -4$ (функция убывает → идём вправо). В точке $w = 5$: $f'(5) = 6$ (функция растёт → идём влево).

### Для функции многих переменных

$$
f(w_1, w_2) = w_1^2 + 3w_2^2
$$

Градиент:

$$
\nabla f = \begin{pmatrix} 2w_1 \\ 6w_2 \end{pmatrix}
$$

В точке $(1, 1)$: $\nabla f = (2, 6)$. Направление спуска — $(-2, -6)$, нормализуем: $(-1/\sqrt{10}, -3/\sqrt{10})$.

---

## 2. Градиентный спуск: базовый алгоритм

### Формула обновления

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \nabla f(\mathbf{w}_t)
$$

где:
- $\eta$ — learning rate (скорость обучения)
- $\nabla f(\mathbf{w}_t)$ — градиент в текущей точке

### Пошаговый алгоритм

```
1. Инициализировать w случайными значениями
2. Вычислить градиент ∇f(w)
3. Обновить w = w - η·∇f(w)
4. Повторять шаги 2-3, пока не сойдётся
```

### Выбор learning rate

| η | Что происходит |
|---|---------------|
| Слишком малый | Медленная сходимость, застревание на плато |
| Оптимальный | Стабильное убывание потерь |
| Слишком большой | Осцилляция, расходимость |
| Очень большой | Взрыв градиентов (NaN) |

Пример: $f(w) = w^2$, минимум в $w = 0$.

- $\eta = 0.1$: $w_0=3 → 2.4 → 1.92 → 1.54 → ... → 0$ — медленно но верно
- $\eta = 1.0$: $w_0=3 → -3 → 3 → -3 → ...$ — осцилляция
- $\eta = 2.0$: $w_0=3 → -9 → 27 → ...$ — расходимость

---

## 3. Стохастический градиентный спуск (SGD)

### Проблема полного градиента

Полный градиент по всему датасету из $N$ примеров:

$$
\nabla L = \frac{1}{N} \sum_{i=1}^{N} \nabla \ell_i
$$

При $N = 1\,000\,000$ — один шаг стоит миллион вычислений.

### SGD: аппроксимация градиента

Берём один случайный пример (или мини-батч размера $B$):

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \nabla \ell_{i_t}(\mathbf{w}_t)
$$

Мини-батч:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \frac{1}{B} \sum_{j=1}^{B} \nabla \ell_{i_j}(\mathbf{w}_t)
$$

### Сравнение

| Метод | На шаг | Точность градиента | Скорость сходимости |
|-------|--------|-------------------|---------------------|
| Full GD | O(N) | Точная | Лучшая на шаг |
| Mini-batch SGD | O(B) | Аппроксимация | Больше шагов, но быстрее по времени |
| SGD (B=1) | O(1) | Очень шумная | Шумная, но быстрая |

Типичный размер батча: 32, 64, 128, 256.

---

## 4. Проблемы оптимизации

### 4.1 Локальные минимумы

Функция может иметь несколько минимумов. Градиентный спуск находит ближайший, не обязательно глобальный.

Для нейросетей: седловых точек гораздо больше, чем локальных минимумов. Это основная проблема, а не минимумы.

### 4.2 Седловые точки

Точка, где градиент нулевой, но это не минимум:

$$
f(x, y) = x^2 - y^2
$$

В точке $(0, 0)$: $\nabla f = (0, 0)$, но по $x$ — минимум, по $y$ — максимум.

### 4.3 Проблема «оврагов» (ill-conditioning)

Когда гессиан $\nabla^2 f$ имеет сильно различные собственные значения, поверхность потерь напоминает узкий овраг. SGD зигзагообразно прыгает между стенками.

Число обусловленности:

$$
\kappa = \frac{\lambda_{max}}{\lambda_{min}}
$$

При $\kappa \gg 1$ — овраг, сходимость медленная.

### 4.4 Исчезающие и взрывающиеся градиенты

В глубоких сетях градиент перемножается на каждой слой:

$$
\frac{\partial L}{\partial \mathbf{w}_1} = \frac{\partial L}{\partial \mathbf{a}_L} \cdot \prod_{l=2}^{L} \frac{\partial \mathbf{a}_l}{\partial \mathbf{a}_{l-1}}
$$

Если каждый множитель $< 1$ — градиент экспоненциально затухает. Если $> 1$ — экспоненциально растёт.

---

## 5. Momentum

### Идея

Добавить «инерцию» — накапливать градиенты прошлых шагов:

$$
\mathbf{v}_t = \gamma \mathbf{v}_{t-1} + \eta \nabla f(\mathbf{w}_t)
$$

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_t
$$

$\gamma \in [0, 1)$ — коэффициент момента (обычно 0.9).

### Зачем

- В направлении длинной оси оврага — момент ускоряет движение
- В направлении короткой оси — осцилляции гасят друг друга
- Помогает «проскакивать» мелкие локальные минимумы

### Nesterov Momentum

Заглядываем на шаг вперёд:

$$
\mathbf{v}_t = \gamma \mathbf{v}_{t-1} + \eta \nabla f(\mathbf{w}_t - \gamma \mathbf{v}_{t-1})
$$

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_t
$$

Вычисляем градиент не в текущей, а в «предсказанной» следующей точке. Даёт лучшую сходимость в теории.

---

## 6. Адаптивные методы

### 6.1 AdaGrad

Накапливаем сумму квадратов градиентов:

$$
\mathbf{s}_t = \mathbf{s}_{t-1} + (\nabla f(\mathbf{w}_t))^2
$$

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \nabla f(\mathbf{w}_t)
$$

Плюс: реже встречающиеся признаки получают больший LR.
Минус: $\mathbf{s}_t$ только растёт → LR монотонно уменьшается → обучение останавливается.

### 6.2 RMSProp

Экспоненциальное скользящее среднее вместо суммы:

$$
\mathbf{s}_t = \beta \mathbf{s}_{t-1} + (1 - \beta)(\nabla f(\mathbf{w}_t))^2
$$

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \nabla f(\mathbf{w}_t)
$$

$\beta$ обычно 0.9. Решает проблему AdaGrad — LR не уменьшается бесконечно.

### 6.3 Adam = Momentum + RMSProp

Первый момент (среднее градиента):

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \nabla f(\mathbf{w}_t)
$$

Второй момент (среднее квадратов):

$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) (\nabla f(\mathbf{w}_t))^2
$$

Коррекция смещения (важно на первых шагах):

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$

Обновление:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
$$

Гиперпараметры по умолчанию: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 0.001$.

### Почему Adam — дефолт

| Оптимизатор | Нужно тюнить LR | Устойчивость | Скорость сходимости |
|-------------|-----------------|--------------|---------------------|
| SGD | Да, чувствителен | Низкая | Медленная без момента |
| SGD+Momentum | LR + γ | Средняя | Хорошая |
| RMSProp | LR + β | Средняя | Хорошая |
| Adam | LR (менее чувствителен) | Высокая | Отличная |

Adam — лучший выбор по умолчанию для большинства задач.

---

## 7. Learning Rate Scheduling

### Почему фиксированный LR — плохо

В начале нужен большой LR для быстрого приближения. В конце — маленький для точной настройки.

### Step Decay

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T \rfloor}
$$

Каждые $T$ эпох уменьшаем LR в $\gamma$ раз. Пример: каждые 10 эпох $\times 0.5$.

### Exponential Decay

$$
\eta_t = \eta_0 \cdot e^{-kt}
$$

Плавное уменьшение. Параметр $k$ контролирует скорость.

### Cosine Annealing

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

Популярен в современных архитектурах (ResNet, ViT). Плавный переход от большого к малому LR.

### Warmup

Начинаем с маленького LR и линейно увеличиваем:

$$
\eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}}, \quad t \leq T_{warmup}
$$

Критически важен для трансформеров и Adam — первые шаги с малым LR предотвращают расходимость.

### One Cycle Policy

1. Warmup: LR растёт от $\eta_{min}$ до $\eta_{max}$
2. Annealing: LR уменьшается обратно до $\eta_{min}$
3. Extra: LR падает ещё ниже

Один цикл — часто достаточно для сходимости. Эмпирически даёт лучшую генерализацию.

---

## 8. Практика

### SGD vs Adam на квадратичной функции

```python
import numpy as np

# Квадратичная функция с оврагом
# f(w) = 0.5 * w^T A w, где A имеет разные собственные значения
A = np.diag([1.0, 50.0])  # овраг: cond = 50

def f(w):
    return 0.5 * w @ A @ w

def grad_f(w):
    return A @ w

w0 = np.array([4.0, 4.0])
lr = 0.01
steps = 200

# Vanilla SGD
w = w0.copy()
path_sgd = [w.copy()]
for _ in range(steps):
    g = grad_f(w)
    w = w - lr * g
    path_sgd.append(w.copy())

# SGD + Momentum
w = w0.copy()
v = np.zeros(2)
gamma = 0.9
path_mom = [w.copy()]
for _ in range(steps):
    g = grad_f(w)
    v = gamma * v + lr * g
    w = w - v
    path_mom.append(w.copy())

# Adam
w = w0.copy()
m = np.zeros(2)
s = np.zeros(2)
beta1, beta2, eps = 0.9, 0.999, 1e-8
path_adam = [w.copy()]
for t in range(1, steps + 1):
    g = grad_f(w)
    m = beta1 * m + (1 - beta1) * g
    s = beta2 * s + (1 - beta2) * g**2
    m_hat = m / (1 - beta1**t)
    s_hat = s / (1 - beta2**t)
    w = w - lr * m_hat / (np.sqrt(s_hat) + eps)
    path_adam.append(w.copy())

print(f"SGD final:    w={path_sgd[-1]}, f={f(path_sgd[-1]):.6f}")
print(f"Momentum final: w={path_mom[-1]}, f={f(path_mom[-1]):.6f}")
print(f"Adam final:   w={path_adam[-1]}, f={f(path_adam[-1]):.6f}")
```

### Adam на реальной задаче

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Генерируем данные
X, y = make_classification(n_samples=1000, n_features=20,
                           n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Стандартизация
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0) + 1e-8
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Логистическая регрессия с Adam
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

w = np.zeros(20)
b = 0.0
m_adam = np.zeros(20)
v_adam = np.zeros(20)
m_bias = 0.0
v_bias = 0.0
beta1, beta2, eps = 0.9, 0.999, 1e-8
lr = 0.01

for t in range(1, 501):
    # Forward
    z = X_train @ w + b
    pred = sigmoid(z)

    # Loss (binary cross-entropy)
    loss = -np.mean(y_train * np.log(pred + 1e-8) +
                    (1 - y_train) * np.log(1 - pred + 1e-8))

    # Backward
    dz = pred - y_train
    dw = X_train.T @ dz / len(y_train)
    db = dz.mean()

    # Adam update
    m_adam = beta1 * m_adam + (1 - beta1) * dw
    v_adam = beta2 * v_adam + (1 - beta2) * dw**2
    m_hat = m_adam / (1 - beta1**t)
    v_hat = v_adam / (1 - beta2**t)
    w -= lr * m_hat / (np.sqrt(v_hat) + eps)

    m_bias = beta1 * m_bias + (1 - beta1) * db
    v_bias = beta2 * v_bias + (1 - beta2) * db**2
    mb_hat = m_bias / (1 - beta1**t)
    vb_hat = v_bias / (1 - beta2**t)
    b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    if t % 100 == 0:
        acc = ((sigmoid(X_test @ w + b) > 0.5) == y_test).mean()
        print(f"Step {t}: loss={loss:.4f}, test_acc={acc:.4f}")
```

### Cosine Annealing визуализация

```python
import numpy as np

def cosine_annealing(t, T, eta_max, eta_min):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(t / T * np.pi))

T = 100
etas = [cosine_annealing(t, T, 0.1, 1e-5) for t in range(T)]
print(f"Start LR: {etas[0]:.5f}")
print(f"Mid LR:   {etas[50]:.5f}")
print(f"End LR:   {etas[-1]:.7f}")
```

---

## Итоги

- **Градиент** — вектор частных производных, указывает направление роста функции
- **SGD** — аппроксимация градиента по мини-батчу, основа обучения нейросетей
- **Momentum** — инерция, помогает в оврагах и ускоряет сходимость
- **Adam** — адаптивный метод (Momentum + RMSProp), лучший выбор по умолчанию
- **LR scheduling** — важнейший гиперпараметр, warmup + cosine annealing — стандарт для LLM
