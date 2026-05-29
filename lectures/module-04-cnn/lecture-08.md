# Лекция 8: Архитектуры CNN — от LeNet до ResNet

## Модуль 4: Сверточные нейронные сети

---

## 1. Эволюция архитектур

Таймлайн ключевых архитектур:

```
1998  LeNet-5        — первая практическая CNN
2012  AlexNet        — прорыв ImageNet (84.7% → 63.3% error)
2014  VGGNet         — простота и глубина
2014  GoogLeNet      — Inception-модуль
2015  ResNet         — residual connections, 152 слоя
2017  DenseNet       — плотные связи
2017  MobileNet      — эффективные сети для мобильных
2020  EfficientNet   — compound scaling
```

Закономерность: чем глубже сеть — тем лучше, но только если решена проблема затухания градиентов.

---

## 2. LeNet-5 (1998)

Первая успешная CNN Яна ЛеКуна для распознавания цифр (MNIST).

**Архитектура:**

```
Вход: 32×32×1
→ C1: Conv 5×5, 6 фильтров → 28×28×6
→ S2: AvgPool 2×2 → 14×14×6
→ C3: Conv 5×5, 16 фильтров → 10×10×16
→ S4: AvgPool 2×2 → 5×5×16
→ C5: Conv 5×5, 120 фильтров → 1×1×120
→ F6: FC 84
→ Выход: FC 10
```

**Подсчёт параметров:**

$$
P_{conv} = K^2 \cdot C_{in} \cdot C_{out} + C_{out}
$$

| Слой | Форма | Параметры |
|------|-------|-----------|
| C1 | 5×5×1×6 | 156 |
| C3 | 5×5×6×16 | 2\,416 |
| C5 | 5×5×16×120 | 48\,120 |
| F6 | 120×84 | 10\,164 |
| Output | 84×10 | 850 |
| **Итого** | | **~61K** |

Для сравнения: FC-сеть на 32×32 → 3M+ параметров. LeNet — в 50 раз меньше.

---

## 3. AlexNet (2012)

Прорыв на ImageNet (top-5 error: 26% → 16.4%). 60M параметров.

**Ключевые инновации:**

1. **ReLU вместо sigmoid/tanh** — в 6 раз быстрее обучение
2. **Dropout (0.5)** — в FC-слоях
3. **GPU-обучение** — два GTX 580
4. **Local Response Normalization** (позже вытеснено BatchNorm)
5. **Data augmentation** — случайные кропы, отражения

**Архитектура (упрощённо):**

```
Вход: 224×224×3
→ Conv 11×11, 96, stride 4 → 55×55×96
→ MaxPool 3×3, stride 2 → 27×27×96
→ Conv 5×5, 256 → 27×27×256
→ MaxPool 3×3, stride 2 → 13×13×256
→ Conv 3×3, 384 → 13×13×384
→ Conv 3×3, 384 → 13×13×384
→ Conv 3×3, 256 → 13×13×256
→ MaxPool 3×3, stride 2 → 6×6×256
→ FC 4096
→ FC 4096
→ FC 1000
```

**Почему 11×11 фильтр в первом слое?** Большое рецептивное поле для захвата крупных признаков при stride=4. Позже отказались в пользу стека 3×3.

---

## 4. VGGNet (2014)

Принцип: **только 3×3 свёртки, только stride=1, только padding=1**.

**Ключевая идея:** стек из двух 3×3 свёрток эквивалентен одной 5×5:

$$
\text{receptive field}(2 \times 3 \times 3) = 5 \times 5
$$

Но параметров меньше:

$$
\begin{aligned}
5 \times 5 &: 5^2 \cdot C^2 = 25C^2 \\
2 \times 3 \times 3 &: 2 \cdot 3^2 \cdot C^2 = 18C^2
\end{aligned}
$$

Экономия 28% параметров + больше нелинейности.

**VGG-16:**

```
Вход: 224×224×3
→ [Conv3-64] × 2 → 224×224×64    → MaxPool → 112×112×64
→ [Conv3-128] × 2 → 112×112×128  → MaxPool → 56×56×128
→ [Conv3-256] × 3 → 56×56×256    → MaxPool → 28×28×256
→ [Conv3-512] × 3 → 28×28×512    → MaxPool → 14×14×512
→ [Conv3-512] × 3 → 14×14×512    → MaxPool → 7×7×512
→ FC 4096 → FC 4096 → FC 1000
```

| Вариант | Слоёв | Параметры |
|---------|-------|-----------|
| VGG-11 | 11 | 133M |
| VGG-13 | 13 | 133M |
| VGG-16 | 16 | 138M |
| VGG-19 | 19 | 144M |

Проблема: огромное количество параметров (138M), из которых 122M — в FC-слоях.

---

## 5. GoogLeNet / Inception (2014)

**Inception-модуль:** параллельная обработка несколькими размерами фильтров.

```
         Вход
      /    |    \     \
   1×1   3×3   5×5  MaxPool
    |   | 1×1 | 1×1 | 1×1
    \   |   |   |   /
       Concat по каналам
```

**Зачем 1×1 свёртка?** Уменьшение размерности (bottleneck):

$$
\text{Без 1×1: } \quad 256 \xrightarrow{3 \times 3 \times 256} 256 \implies 590K \text{ params}
$$

$$
\text{С 1×1 bottleneck: } \quad 256 \xrightarrow{1 \times 1} 64 \xrightarrow{3 \times 3 \times 64} 64 \xrightarrow{1 \times 1} 256 \implies 70K \text{ params}
$$

Уменьшение в **8.4 раза** при сопоставимой выразительности.

**Inception-v1 (GoogLeNet):** 22 слоя, всего 6.8M параметров (в 20 раз меньше VGG-16).

**Эволюция Inception:**

| Версия | Инновация | Параметры |
|--------|-----------|-----------|
| v1 | Bottleneck 1×1 | 6.8M |
| v2 | Factorization: 5×5 → 2×(3×3), BatchNorm | 11.2M |
| v3 | 7×7 → (1×7)+(7×1), RMSProp | 23.8M |
| v4 | Residual connections + Inception | 42.7M |

---

## 6. ResNet (2015) — самая важная архитектура

### Проблема глубоких сетей

56-слойная сеть обучалась **хуже** 20-слойной. Не overfitting — training error тоже выше.

**Причина:** глубокую сеть оптимизировать сложнее. Добавление слоёв должно как минимум не ухудшать результат (identity mapping), но SGD не находит этого решения.

### Residual learning

Вместо обучения $\mathcal{H}(x)$ — учим **остаток** (residual):

$$
\mathcal{F}(x) = \mathcal{H}(x) - x \implies \mathcal{H}(x) = \mathcal{F}(x) + x
$$

```
     x ──────────────────────→ (+) → output
     │                            ↑
     └→ Conv → ReLU → Conv → ReLU┘
          F(x)
```

**Почему это работает:**

1. Если слой не нужен — $\mathcal{F}(x) \to 0$, получается identity mapping
2. Градиент течёт через skip connection: $\frac{\partial}{\partial x}(F(x) + x) = \frac{\partial F}{\partial x} + 1$
3. Единица гарантирует, что градиент не исчезнет полностью

### Residual Block

**Базовый блок (ResNet-18/34):**

$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

где $\mathcal{F}$ — два Conv 3×3 с BatchNorm и ReLU.

**Bottleneck блок (ResNet-50/101/152):**

```
x → 1×1 Conv (↓dim) → 3×3 Conv → 1×1 Conv (↑dim) → + x
     256→64              64→64      64→256
```

Параметры bottleneck: $1 \times 256 \times 64 + 3 \times 64 \times 64 + 1 \times 64 \times 256 = 69\,632$

Без bottleneck (два 3×3 на 256): $2 \times 9 \times 256^2 = 1\,179\,648$

Экономия в **17 раз**.

### Семейство ResNet

| Модель | Слоёв | Параметры | Top-5 ImageNet |
|--------|-------|-----------|----------------|
| ResNet-18 | 18 | 11.7M | 89.1% |
| ResNet-34 | 34 | 21.8M | 91.4% |
| ResNet-50 | 50 | 25.6M | 92.9% |
| ResNet-101 | 101 | 44.5M | 93.7% |
| ResNet-152 | 152 | 60.2M | 94.1% |

ResNet-152 обучается лучше VGG-16 при **в 2 раза меньшем** числе параметров.

### Variant: ResNeXt

$$
\mathcal{F}(x) = \sum_{i=1}^{C} \mathcal{T}_i(x)
$$

где $C$ — кардинальность (число параллельных путей). При одинаковом числе параметров ResNeXt-50 ($C=32$) даёт +1% accuracy.

---

## 7. DenseNet (2017)

Каждый слой получает на вход **все** предыдущие feature maps:

$$
x_l = H_l([x_0, x_1, \ldots, x_{l-1}])
$$

где $[\cdot]$ — конкатенация.

```
x₀ → H₁ → x₁ → H₂ → x₂ → H₃ → x₃
 ↓     ↓      ↓      ↓
 └───→[x₀,x₁,x₂,x₃]→ H₄ → ...
```

**Dense Block:** рост каналов на $k$ (growth rate) на каждом слое.

Параметры: DenseNet-121 — 8M (в 3 раза меньше ResNet-50), accuracy сопоставима.

**Проблема:** конкатенация → быстро растёт потребление памяти. Требуется transition layer (1×1 Conv + AvgPool) между dense blocks.

---

## 8. MobileNet (2017) — эффективные сети

### Depthwise Separable Convolution

Стандартная свёртка: $D_K \times D_K \times M \times N$

Depthwise separable: Depthwise ($D_K \times D_K \times 1 \times M$) + Pointwise ($1 \times 1 \times M \times N$)

**Коэффициент сжатия:**

$$
\frac{D_K^2 \cdot M + M \cdot N}{D_K^2 \cdot M \cdot N} = \frac{1}{N} + \frac{1}{D_K^2}
$$

Для $D_K=3$, $N=256$: в **8-9 раз** меньше вычислений.

| Модель | MADD | Параметры | Top-1 |
|--------|------|-----------|-------|
| MobileNet-v1 | 569M | 4.2M | 70.6% |
| MobileNet-v2 | 300M | 3.4M | 72.0% |
| ResNet-50 | 4100M | 25.6M | 76.0% |

MobileNet в **7-13 раз** легче, всего на 4-6% хуже.

### MobileNet-v2: Inverted Residual

```
     x (low-dim)
     → 1×1 expand (↑dim)
     → 3×3 depthwise
     → 1×1 project (↓dim)
     → + x (skip connection)
```

Linear bottleneck (без ReLU в последнем 1×1) — ReLU уничтожает информацию в низкоразмерном пространстве.

---

## 9. EfficientNet (2020)

**Compound Scaling** — одновременное масштабирование трёх измерений:

$$
\begin{aligned}
\text{depth:} \quad & d = \alpha^\phi \\
\text{width:} \quad & w = \beta^\phi \\
\text{resolution:} \quad & r = \gamma^\phi
\end{aligned}
$$

при ограничении:

$$
\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
$$

где $\phi$ — коэффициент масштабирования, $\alpha, \beta, \gamma$ — константы (найдены через grid search).

**Результат:** EfficientNet-B0 (5.3M параметров) обходит ResNet-152 (60M) на ImageNet.

| Модель | Параметры | Top-1 |
|--------|-----------|-------|
| EfficientNet-B0 | 5.3M | 77.1% |
| EfficientNet-B3 | 12M | 81.6% |
| EfficientNet-B7 | 66M | 84.3% |

---

## 10. Transfer Learning

### Идея

Обученная на ImageNet сеть — универсальный экстрактор признаков. Ранние слои выделяют границы и текстуры (универсальные для любого зрения), глубокие — специфичные для dataset классы.

```
┌─────────────────────────────────────┐
│  ImageNet (1.2M изображений)        │
│                                     │
│  Layer 1:  границы, градиенты  ─────┼──→ замораживаем
│  Layer 2:  текстуры, паттерны   ─────┼──→ замораживаем
│  Layer 3:  части объектов        ─────┼──→ замораживаем (или fine-tune)
│  Layer 4:  объекты               ─────┼──→ fine-tune
│  FC:       1000 классов          ─────┼──→ заменяем на свои классы
└─────────────────────────────────────┘
```

### Три стратегии

| Стратегия | Что делаем | Когда | Данных нужно |
|-----------|-----------|-------|-------------|
| Feature extraction | Заморозить всё, заменить FC | Мало данных (< 1K) | Мало |
| Fine-tuning последних | Разморозить 2-3 слоя + FC | Средне (1K-10K) | Средне |
| Полный fine-tune | Разморозить всё | Много (> 10K) | Много |

### Почему это работает

CNN выстраивает иерархию фильтров:

```
Пиксели → Границы → Текстуры → Части → Объекты → Классы
  (L1)      (L2)      (L3)      (L4)     (L5)      (FC)
```

Границы и текстуры — одинаковы для кошек, машин, медицинских снимков. Поэтому веса transfer'ятся между доменами.

### Код: Transfer Learning на ResNet

```python
import torchvision.models as models

# Загружаем предобученную ResNet-18
model = models.resnet18(pretrained=True)

# Стратегия 1: Feature Extraction
for param in model.parameters():
    param.requires_grad = False  # замораживаем всё

# Заменяем последний слой под свои классы
num_features = model.fc.in_features  # 512
model.fc = nn.Linear(num_features, 10)  # 10 своих классов

# Обучаем только последний слой
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
# → ~512×10 = 5120 параметров вместо 11M
```

```python
# Стратегия 2: Fine-tuning последних слоёв
for param in model.parameters():
    param.requires_grad = False

# Размораживаем layer4 + FC
for param in model.layer4.parameters():
    param.requires_grad = True
    param.data *= 0.01  # уменьшаем веса для стабильности

model.fc = nn.Linear(num_features, 10)
optimizer = torch.optim.Adam(
    list(model.layer4.parameters()) + list(model.fc.parameters()),
    lr=0.0001  # ниже LR для предобученных слоёв
)
```

---

## 11. Практика

### Реализация ResNet Block на PyTorch

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet basic block (для ResNet-18/34)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # residual connection
        out = torch.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet bottleneck block (для ResNet-50/101/152)"""
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


# Сборка ResNet-18
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)
```

### Сравнение архитектур на CIFAR-10

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Данные
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform)
trainloader = DataLoader(trainset, batch_size=128,
                         shuffle=True, num_workers=2)

# Обучение
import torch.optim as optim

model = ResNet18(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(50):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        acc = 100. * correct / total
        print(f'Epoch {epoch+1}: loss={total_loss/len(trainloader):.4f}, '
              f'acc={acc:.2f}%')
```

### Подсчёт параметров архитектур

```python
def count_params(model):
    """Подсчёт обучаемых параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Сравнение
models = {
    'ResNet-18 (CIFAR)': ResNet18(num_classes=10),
}

for name, model in models.items():
    print(f'{name}: {count_params(model):,} параметров')
```

---

## Итоги лекции

**Ключевые принципы развития CNN:**

1. **VGG:** uniform 3×3 — простота и глубина
2. **Inception:** multi-scale processing + 1×1 bottleneck
3. **ResNet:** skip connections — обучаем сотни слоёв
4. **MobileNet:** depthwise separable — эффективный inference
5. **EfficientNet:** compound scaling — баланс точности и скорости

**CNN — иерархия фильтров:** от пикселей к границам, от границ к текстурам, от текстур к частям объектов, от частей — к целым объектам. Каждый слой строится на результате предыдущего.

**Главный урок:** ResNet — самая влиятельная архитектура. Skip connections используются повсюду: трансформеры, U-Net, GAN, диффузионные модели. Transfer learning делает предобученные CNN универсальным фундаментом для любого визуального проекта.
