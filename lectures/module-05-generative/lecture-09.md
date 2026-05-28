# Лекция 9: Генеративно-состязательные сети (GAN)

## Модуль 5: Генеративные сети

---

## 9.1 Задача генерации

До GAN нейросети умели только **различать** (дискриминировать):
- Классификация: $x \to y$
- Регрессия: $x \to \hat{y}$

Генерация — обратная задача: $z \to x$, где $z$ — случайный шум, $x$ — реалистичный объект.

**Типы генеративных моделей:**

| Модель | Принцип | Сложность обучения |
|--------|---------|--------------------|
| GAN | Состязательная игра | Нестабильная |
| VAE | Вариационный вывод | Размытый выход |
| Flow | Обратимое преобразование | Дорого вычислять |
| Diffusion | Постепенный деноизинг | Медленная генерация |

---

## 9.2 Архитектура GAN

GAN состоит из двух сетей:

**Генератор $G(z, \theta_g)$** — создаёт фейковые данные из шума $z \sim p_z$

**Дискриминатор $D(x, \theta_d)$** — отличает реальные данные от фейковых

$$
D: X \to [0, 1] \quad \text{(вероятность "реальное")}
$$

$$
G: Z \to X \quad \text{(шум} \to \text{"реалистичное" изображение)}
$$

**Интуиция:** фальшивомонетчик vs полицейский. Оба учатся, соревнуясь.

---

## 9.3 Минимаксная игра

Оригинальная функция потерь (Goodfellow et al., 2014):

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

### Оптимальный дискриминатор

При фиксированном $G$:

$$
D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}
$$

Если $p_g = p_{data}$, то $D^*(x) = 0.5$ — дискриминатор угадывает как монетку.

### Глобальный оптимум

$V(G^*) = -\log 4$ достигается при $p_g = p_{data}$.

**Доказательство (идея):**

Подставляем $D^*_G$ в $V$:

$$
C(G) = -\log 4 + 2 \cdot JSD(p_{data} \| p_g)
$$

где $JSD$ — расстояние Йенсена-Шеннона:

$$
JSD(P \| Q) = \frac{1}{2} KL\left(P \| \frac{P+Q}{2}\right) + \frac{1}{2} KL\left(Q \| \frac{P+Q}{2}\right)
$$

$JSD \geq 0$ с равенством при $P = Q$. Значит минимум $C(G) = -\log 4$.

---

## 9.4 Обучение GAN на практике

### Шаг обучения

1. **Обучаем $D$** (k шагов, обычно k=1):

$$
\max_D \; \frac{1}{m}\sum_{i=1}^{m}\left[\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))\right]
$$

2. **Обучаем $G$** (1 шаг):

$$
\min_G \; \frac{1}{m}\sum_{i=1}^{m} \log(1 - D(G(z^{(i)})))
$$

### Trick: альтернативная функция потерь генератора

$\log(1 - D(G(z)))$ даёт слабый градиент, когда $D$ хорошо отличает фейки. Решение:

$$
\max_G \; \mathbb{E}_z[\log D(G(z))]
$$

Вместо минимизации вероятности "фейк" — максимизация вероятности "реальное".

### Размерность шума

Обычно $z \sim \mathcal{N}(0, I)$ с размерностью 100-128. Латентное пространство должно быть:
- Непрерывным (малое изменение $z$ → малое изменение $G(z)$)
- Покрывающим (любой $x$ достижим из некоторого $z$)

---

## 9.5 Проблемы обучения GAN

### 1. Mode collapse

Генератор находит один "проходной" тип выхода и генерирует только его.

**Пример:** GAN для MNIST генерирует только "1", потому что дискриминатору сложнее отличить "1" от реальной "1".

**Решения:** Minibatch discrimination, Unrolled GAN, WGANGP.

### 2. Исчезающий градиент

Если $D$ слишком хорош, $\nabla_G \to 0$ и $G$ перестаёт учиться.

### 3. Осцилляции

$G$ и $D$ могут циклично менять стратегии, не сходясь.

### 4. Оценка качества

Нет прямой метрики качества генерации. Приближения:
- **Inception Score (IS):** $\exp(\mathbb{E}_x KL(p(y|x) \| p(y)))$ — высокое при чётких, разнообразных генерациях
- **FID (Fréchet Inception Distance):** сравнение распределений фичей Inception-v3

$$
FID = \|\mu_r - \mu_g\|^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

Ниже FID → лучше качество. State-of-the-art: FID < 2 на CIFAR-10.

---

## 9.6 DCGAN

Deep Convolutional GAN (Radford et al., 2015) — первые стабильные архитектурные правила:

**Архитектурные принципы:**
1. Заменить pooling страйдированной свёрткой (D) и транспонированной свёрткой (G)
2. BatchNorm везде кроме выхода $G$ и входа $D$
3. Убрать полносвязные слои (только conv)
4. ReLU в $G$, LeakyReLU в $D$

**Генератор DCGAN (для 64×64):**

```
z (100,) → reshape (100, 1, 1)
    → ConvTranspose2d(100, 512, 4, 1, 0) → BN → ReLU    → (512, 4, 4)
    → ConvTranspose2d(512, 256, 4, 2, 1) → BN → ReLU    → (256, 8, 8)
    → ConvTranspose2d(256, 128, 4, 2, 1) → BN → ReLU    → (128, 16, 16)
    → ConvTranspose2d(128, 64, 4, 2, 1)  → BN → ReLU    → (64, 32, 32)
    → ConvTranspose2d(64, 3, 4, 2, 1)    → Tanh          → (3, 64, 64)
```

**Подсчёт параметров генератора:**

$$
100 \times 512 \times 4 \times 4 + 512 \times 256 \times 4 \times 4 + \ldots \approx 3.5M
$$

---

## 9.7 WGAN — Wasserstein GAN

**Проблема оригинального GAN:** JS-дивергенция не непрерывна относительно параметров $\theta_g$, когда носители $p_g$ и $p_{data}$ не пересекаются (что типично в пространстве высоких размерностей).

**Решение (Arjovsky et al., 2017):** Earth Mover's (Wasserstein) расстояние:

$$
W(p_{data}, p_g) = \inf_{\gamma \in \Pi(p_{data}, p_g)} \mathbb{E}_{(x,y)\sim\gamma}[\|x - y\|]
$$

По Канторовичу-Рубинштейну:

$$
W(p_{data}, p_g) = \sup_{\|f\|_L \leq 1} \left\{\mathbb{E}_{x \sim p_{data}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]\right\}
$$

где $\|f\|_L \leq 1$ — 1-Липшицевы функции.

### WGAN loss

Заменяем дискриминатор на **критика** $f_w$:

$$
\mathcal{L}_D = -\mathbb{E}_{x \sim p_{data}}[f_w(x)] + \mathbb{E}_{z \sim p_z}[f_w(G(z))]
$$

$$
\mathcal{L}_G = -\mathbb{E}_{z \sim p_z}[f_w(G(z))]
$$

### Липшицево ограничение

Два способа:
1. **Weight clipping** (WGAN): $w \leftarrow \text{clip}(w, -c, c)$ — просто, но ограничивает ёмкость
2. **Gradient penalty** (WGAN-GP):

$$
GP = \lambda \cdot \mathbb{E}_{\hat{x}}\left[(\|\nabla_{\hat{x}} f_w(\hat{x})\|_2 - 1)^2\right]
$$

где $\hat{x} = \epsilon x + (1-\epsilon) G(z)$ — точки на интерполяции.

---

## 9.8 Conditional GAN (cGAN)

Добавляем conditioning — метку класса $y$:

$$
\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}[\log D(x|y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z|y)|y))]
$$

**Как подавать условие:**
- Конкатенация к входу: $D(x, y)$, $G(z, y)$
- Projection: $y^T V D_{last}(x)$ (Projection Discriminator, Miyato & Koyama 2018)
- Embedding + сложение с фичами

**Применения:**
- Pix2Pix: $y$ = входное изображение (edge map → photo)
- CycleGAN: перевод между доменами без парных данных
- StyleGAN: $y$ = стилевые параметры через ADA

---

## 9.9 StyleGAN

StyleGAN (Karras et al., 2019) — state-of-the-art для генерации лиц.

### Ключевые идеи

1. **Mapping network** $f: \mathcal{Z} \to \mathcal{W}$ — 8 MLP слоёв

$$
w = f(z), \quad w \in \mathcal{W}, \; \dim(\mathcal{W}) = 512
$$

Промежуточное пространство $\mathcal{W}$ менее запутанное (less entangled).

2. **AdaIN** (Adaptive Instance Normalization) — подача стиля:

$$
\text{AdaIN}(x_i, y_i) = y_{s,i} \cdot \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}
$$

где $y_{s,i} = A_s(w)$, $y_{b,i} = A_b(w)$ — аффинные преобразования от $w$.

3. **Stochastic variation** — шум на каждом слое для деталей (веснушки, волосы)

4. **Style mixing** — разные $w$ для грубых ($4 \times 4 \ldots 8 \times 8$) и мелких ($64 \times 64 \ldots 1024 \times 1024$) разрешений.

### StyleGAN2

Улучшения:
- Weight demodulation вместо AdaIN (убирает капли-артефакты)
- Lazy regularization (R1 раз в 16 шагов)
- Path length regularization

### Метрика FID StyleGAN2

| Датасет | Разрешение | FID |
|---------|------------|-----|
| FFHQ | 1024×1024 | 2.84 |
| LSUN Car | 512×384 | 3.27 |
| CIFAR-10 | 32×32 | 2.42 |

---

## 9.10 Progressive Growing

StyleGAN обучается от малого разрешения к большему:

1. Начинаем с $4 \times 4$
2. Обучаем до сходимости
3. Добавляем слой $8 \times 8$ с плавным fade-in:

$$
x_{out} = \alpha \cdot x_{new} + (1 - \alpha) \cdot \text{upsample}(x_{prev})
$$

4. Повторяем до $1024 \times 1024$

**Почему это работает:** каждый этап решает простую задачу — грубые формы, затем детали.

---

## 9.11 Практика

### DCGAN на CIFAR-10 (PyTorch)

```python
import torch
import torch.nn as nn

# --- Генератор ---
class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # z_dim x 1 x 1 → ngf*8 x 4 x 4
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # → ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # → ngf*2 x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # → nc x 32 x 32
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

# --- Дискриминатор ---
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # nc x 32 x 32 → ndf x 16 x 16
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → ndf*2 x 8 x 8
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # → ndf*4 x 4 x 4
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # → 1 x 1 x 1
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)
```

### Цикл обучения

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

fixed_z = torch.randn(64, 100, 1, 1, device=device)

for epoch in range(200):
    for real, _ in dataloader:
        real = real.to(device)
        bs = real.size(0)

        # --- Обучаем D ---
        D.zero_grad()

        # Реальные → метка 1
        label_real = torch.ones(bs, device=device)
        d_real = D(real)
        loss_d_real = criterion(d_real, label_real)

        # Фейковые → метка 0
        z = torch.randn(bs, 100, 1, 1, device=device)
        fake = G(z).detach()
        label_fake = torch.zeros(bs, device=device)
        d_fake = D(fake)
        loss_d_fake = criterion(d_fake, label_fake)

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        opt_D.step()

        # --- Обучаем G ---
        G.zero_grad()
        z = torch.randn(bs, 100, 1, 1, device=device)
        fake = G(z)
        # Хотим обмануть D → метка 1
        label_g = torch.ones(bs, device=device)
        d_fake = D(fake)
        loss_g = criterion(d_fake, label_g)
        loss_g.backward()
        opt_G.step()

    print(f"Epoch {epoch}: D_loss={loss_d.item():.4f}, G_loss={loss_g.item():.4f}")
```

### WGAN-GP критик (для сравнения)

```python
def gradient_penalty(critic, real, fake, device):
    bs = real.size(0)
    eps = torch.rand(bs, 1, 1, 1, device=device)
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)

    d_interp = critic(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(bs, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# Цикл: D обновляется 5 раз на каждый шаг G
# loss_D = -d_real.mean() + d_fake.mean() + 10 * gradient_penalty(...)
# loss_G = -critic(G(z)).mean()
```

---

## Резюме

| Концепция | Ключевая формула | Что даёт |
|-----------|-------------------|----------|
| GAN | $\min_G \max_D V(D,G)$ | Фреймворк генерации |
| Оптимальный D | $D^* = \frac{p_{data}}{p_{data}+p_g}$ | Теоретический оптимум |
| DCGAN | Conv + BN + ReLU/LeakyReLU | Стабильная архитектура |
| WGAN | Wasserstein distance | Осмысленный loss |
| WGAN-GP | $\|\nabla f\|_2 \approx 1$ | Стабильное Липшицево ограничение |
| cGAN | Conditioning на $y$ | Контролируемая генерация |
| StyleGAN | $\mathcal{Z} \to \mathcal{W}$ + AdaIN | SOTA качество |
| FID | $FID = \|\Delta\mu\|^2 + Tr(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})$ | Оценка качества |

---

## Литература

1. Goodfellow et al. "Generative Adversarial Nets" (2014) — оригинальная статья
2. Radford et al. "Unsupervised Representation Learning with DCGANs" (2015)
3. Arjovsky et al. "Wasserstein GAN" (2017)
4. Gulrajani et al. "Improved Training of WGANs" (2017) — WGAN-GP
5. Mirza & Osindero "Conditional GANs" (2014)
6. Karras et al. "A Style-Based Generator Architecture for GANs" (2019)
7. Karras et al. "Analyzing and Improving the Image Quality of StyleGAN" (2020)
