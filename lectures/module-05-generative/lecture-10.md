# Лекция 10: Диффузионные модели и VAE

> Модуль 5: Генеративные сети | Лекция 10

---

## 10.1 Задача генеративного моделирования

Мы хотим обучить модель, которая **генерирует новые данные**, похожие на тренировочные.

### Три подхода

| Модель | Принцип | Плюс | Минус |
|--------|---------|------|-------|
| GAN | Адверсарная игра | Резкие картинки | Нестабильность, mode collapse |
| VAE | Вариационный вывод | Стабильность, латентное пространство | Размытые картинки |
| Diffusion | Итеративная денойзация | Лучшее качество, diversity | Медленная генерация |

### Формальная постановка

Дано: выборка $\{x_1, \ldots, x_N\}$ из неизвестного распределения $p_{\text{data}}(x)$.

Найти: модель $p_\theta(x)$, аппроксимирующую $p_{\text{data}}(x)$.

**Проблема**: $p_\theta(x)$ в的高оразмерном пространстве — невозможно моделировать напрямую.

---

## 10.2 Вариационный автокодировщик (VAE)

### Автокодировщик — напоминание

Обычный AE:
- Энкодер: $z = f_\phi(x)$ — сжимает в латентное пространство
- Декодер: $\hat{x} = g_\theta(z)$ — восстанавливает

Проблема: латентное пространство неструктурировано → нельзя семплировать.

### Вариационный подход

Вместо точки $z$ — **распределение**:

$$
q_\phi(z \mid x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x) \cdot I)
$$

Энкодер выдаёт $\mu$ и $\sigma$ — параметры нормального распределения.

### Репараметризационный трюк

Нельзя бэкпропить через случайность напрямую. Трюк:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Теперь $z$ — дифференцируемая функция от $\mu$, $\sigma$ и $\epsilon$.

### ELBO — нижняя граница правдоподобия

Логарифм правдоподобия:

$$
\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{реконструкция}} - \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{регуляризация}}
$$

Это **ELBO** (Evidence Lower Bound). Максимизируя его — приближаем правдоподобие.

### KL-дивергенция для двух гауссианов

При $p(z) = \mathcal{N}(0, I)$:

$$
D_{\text{KL}}(q_\phi(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^{J}\left(\mu_j^2 + \sigma_j^2 - \ln \sigma_j^2 - 1\right)
$$

Это штраф за отклонение $q_\phi$ от стандартного нормального.

### Функция потерь VAE

$$
\mathcal{L}_{\text{VAE}} = \underbrace{\|x - \hat{x}\|^2}_{\text{MSE реконструкция}} + \beta \cdot \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{регуляризация}}
$$

При $\beta > 1$ — **$\beta$-VAE**: сильнее структурирует латентное пространство, но хуже реконструкция.

### Архитектура VAE

```
x → [Encoder] → μ, σ → z = μ + σ·ε → [Decoder] → x̂
                 ↑
           репараметризация
```

Размерности:
- Вход: $x \in \mathbb{R}^{784}$ (MNIST)
- Латентное: $z \in \mathbb{R}^{20}$
- Выход: $\hat{x} \in \mathbb{R}^{784}$

### Латентное пространство

Если KL хорошо оптимизирован — латентное пространство **непрерывное**:
- Можно интерполировать: $z = (1-t)z_1 + tz_2$, $t \in [0, 1]$
- Можно арифметику: $z_{\text{улыбка}} = z_{\text{улыбающий}} - z_{\text{нейтральный}}$

### Практика: VAE на PyTorch

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def loss_function(self, x, x_recon, mu, logvar):
        recon = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kl
```

---

## 10.3 Диффузионные модели: Forward Process

### Интуиция

Постепенно добавляем шум к картинке, пока она не станет чистым гауссовым шумом. Потом учим сеть **обращать** этот процесс.

### Прямой процесс (диффузия)

За $T$ шагов добавляем гауссов шум:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

где $\beta_t$ — variance schedule (дисперсия шума на шаге $t$).

### Ключевое свойство: прямой переход от $x_0$ к $x_t$

Обозначим $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$.

Тогда:

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$

**Семплирование за один шаг**:

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Это **ключевая формула** — позволяет сразу получить зашумлённую версию на любом шаге $t$.

### Variance schedule

| Schedule | Формула | Свойство |
|----------|---------|----------|
| Linear | $\beta_t$ от $\beta_1$ до $\beta_T$ | Простая |
| Cosine | $\bar{\alpha}_t = \frac{f(t)}{f(0)}$, $f(t) = \cos^2\frac{t/T + s}{1 + s}\frac{\pi}{2}$ | Лучше для изображений |
| Sigmoid | $\beta_t = \text{sigmoid}(...)$ | Гибкая |

---

## 10.4 Reverse Process (Денойзинг)

### Обратный процесс

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
$$

Нейросеть предсказывает **среднее** $\mu_\theta$ зашумлённого изображения.

### Параметризация через предсказание шума

Вместо $\mu_\theta$ — предсказываем **сам шум** $\epsilon_\theta(x_t, t)$:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

Нейросеть $\epsilon_\theta$ (обычно U-Net) получает $(x_t, t)$ и предсказывает шум $\epsilon$.

### Упрощённая функция потерь (DDPM)

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

где:
- $t \sim \text{Uniform}\{1, \ldots, T\}$ — случайный шаг
- $\epsilon \sim \mathcal{N}(0, I)$ — истинный шум
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

**Алгоритм обучения**:
1. Берём $x_0$ из данных
2. Семплируем $t \sim \text{Uniform}(1, T)$
3. Семплируем $\epsilon \sim \mathcal{N}(0, I)$
4. Вычисляем $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
5. $\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$
6. Шаг градиентного спуска

### Алгоритм генерации (семплирование)

```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    z ~ N(0, I)  if t > 1, else z = 0
    x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-ᾱ_t)) * ε_θ(x_t, t)) + σ_t * z
return x_0
```

$T$ шагов нейросети → медленно, но качественно.

---

## 10.5 U-Net — архитектура денойзера

### Структура

```
Вход: x_t (H×W×C) + t (timestep embedding)
    │
    ↓ [Conv blocks] → [Downsample] × N     ← encoder
    │                                         ↓
    │                              [Bottleneck, attention]
    │                                         ↑
    ↓ [Conv blocks] + [Skip connections] ← [Upsample] × N   ← decoder
    │
    Выход: ε_θ(x_t, t) — предсказанный шум (H×W×C)
```

### Timestep embedding

Время $t$ кодируется через **позиционное кодирование**:

$$
\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)
$$

Подается в каждый residual block через AdaGN (Adaptive Group Norm).

### Attention

Self-attention на разрешениях 16×16, 8×8 — связывает далёкие пиксели.

---

## 10.6 Ускорение генерации

### Проблема DDPM

$T = 1000$ шагов → 1000 прогонов U-Net → **медленно**.

### DDIM (Denoising Diffusion Implicit Models)

Детерминистический вариант:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}}_{\text{предсказанный } x_0} + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta
$$

Можно использовать **подмножество** шагов: $T' = 50$ вместо $1000$.

### Classifier-Free Guidance (CFG)

Обучаем модель **с и без** условия (class label / text prompt).

Во время генерации:

$$
\tilde{\epsilon}_\theta = (1 + w) \cdot \epsilon_\theta(x_t, c) - w \cdot \epsilon_\theta(x_t, \emptyset)
$$

где $w$ — guidance scale ($w = 3 \ldots 7.5$ типично).

Больше $w$ → лучше качество, меньше разнообразие.

---

## 10.7 Conditional Generation

### Класс-обусловленная генерация

Условие $c$ (метка класса) подаётся через:
- **Embedding** + сложение с timestep embedding
- **Cross-attention** с feature maps

### Текст-обусловленная генерация (Stable Diffusion)

Текст → CLIP encoder → sequence of embeddings → cross-attention в U-Net.

### Latent Diffusion (Stable Diffusion)

Вместо пикселей — работаем в **латентном пространстве** VAE:

1. **VAE encoder**: $x \in \mathbb{R}^{3 \times 512 \times 512}$ → $z \in \mathbb{R}^{4 \times 64 \times 64}$ (в 48 раз меньше!)
2. **Diffusion** работает с $z$ — гораздо быстрее
3. **VAE decoder**: $z$ → $x$

Это делает генерацию изображений возможной на потребительских GPU.

---

## 10.8 Сравнение VAE и Diffusion

| Свойство | VAE | Diffusion (DDPM) |
|----------|-----|-------------------|
| Число шагов генерации | 1 (прямой проход) | $T = 1000$ |
| Качество | Размытое | Резкое, SOTA |
| Стабильность обучения | Высокая | Высокая |
| Латентное пространство | Структурированное | Неявное |
| Разнообразие | Ограничено KL | Высокое |
| Теоретическая основа | ELBO | Вариационный вывод |
| Скорость генерации | Быстрая | Медленная (ускоряется DDIM) |

---

## 10.9 Практика: DDPM на MNIST

```python
import torch
import torch.nn as nn
import math

# --- Variance schedule ---
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

T = 300
betas = linear_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# --- Forward process ---
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    return sqrt_alpha * x_0 + sqrt_one_minus * noise

# --- Simple U-Net ---
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(1, 32)
        self.down2 = DoubleConv(32, 64)
        self.bot = DoubleConv(64, 128)
        self.up2 = DoubleConv(128 + 64, 64)
        self.up1 = DoubleConv(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Timestep MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
        )
    
    def forward(self, x, t):
        # Timestep embedding
        t_emb = self.time_mlp(t.unsqueeze(-1).float())  # (B, 32)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # (B, 32, 1, 1)
        
        # Encoder
        d1 = self.down1(x) + t_emb  # (B, 32, 28, 28)
        d2 = self.down2(self.pool(d1))  # (B, 64, 14, 14)
        
        # Bottleneck
        b = self.bot(self.pool(d2))  # (B, 128, 7, 7)
        
        # Decoder
        u2 = self.up(b)
        u2 = self.up2(torch.cat([u2, d2], dim=1))  # (B, 64, 14, 14)
        u1 = self.up(u2)
        u1 = self.up1(torch.cat([u1, d1], dim=1))  # (B, 32, 28, 28)
        
        return self.out(u1)

# --- Training loop ---
model = SimpleUNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for x, _ in train_loader:  # MNIST
        x = x  # (B, 1, 28, 28)
        t = torch.randint(0, T, (x.size(0),))
        noise = torch.randn_like(x)
        x_noisy = q_sample(x, t, noise)
        pred_noise = model(x_noisy, t)
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- Sampling ---
@torch.no_grad()
def sample(model, shape):
    x = torch.randn(shape)
    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, dtype=torch.long)
        pred_noise = model(x, t_batch)
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        alpha_cumprod_prev = alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
        
        beta_t = 1 - alpha_t
        
        # Predicted mean
        x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        x0_pred = x0_pred.clamp(-1, 1)
        
        mean = (1 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * pred_noise)
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * noise
        else:
            x = mean
    return x
```

---

## 10.10 Практика: VAE на MNIST

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
        )
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.enc(x.view(-1, 784))
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar
    
    def loss(self, x, recon, mu, logvar, beta=1.0):
        bce = nn.functional.binary_cross_entropy(recon, x.view(-1, 784), reduction='sum')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + beta * kl

# Обучение
vae = VAE(latent_dim=2)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(20):
    for x, _ in train_loader:
        recon, mu, logvar = vae(x)
        loss = vae.loss(x, recon, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()

# Визуализация латентного пространства (latent_dim=2)
with torch.no_grad():
    z = torch.randn(64, 2)
    samples = vae.dec(z).view(-1, 28, 28)
```

---

## Резюме

| Концепция | Формула | Суть |
|-----------|---------|------|
| ELBO | $\mathbb{E}[\log p(x\|z)] - D_{\text{KL}}(q\|p)$ | Нижняя граница правдоподобия |
| Репараметризация | $z = \mu + \sigma \cdot \epsilon$ | Бэкпроп через стохастичность |
| Forward diffusion | $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$ | Зашумление за один шаг |
| DDPM loss | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ | Предсказание шума |
| CFG | $(1+w)\epsilon_\theta(c) - w\epsilon_\theta(\emptyset)$ | Условная генерация |
