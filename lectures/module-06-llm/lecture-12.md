# Лекция 12: Обучение больших языковых моделей

## Модуль 6: LLM | Лекция 12 из 12

---

## 1. Что такое языковая модель

Языковая модель — это распределение вероятностей над последовательностями токенов:

$$
P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{<t})
$$

**Цель**: для каждого следующего токена предсказать распределение по всему словарю.

### Словарь и токенизация

BPE (Byte Pair Encoding) — итеративное слияние частых пар:

```
"lower" → ["l", "o", "w", "e", "r"]
         → ["lo", "w", "er"]      (слияние l+o, e+r)
         → ["low", "er"]           (слияние lo+w)
```

Типичные размеры словаря: 32K–128K токенов.

---

## 2. Функция потерь: кросс-энтропия

### Определение

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(w_t \mid w_{<t}; \theta)
$$

Это **кросс-энтропия** между истинным распределением (one-hot) и предсказанием модели:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \sum_{v=1}^{V} y_{t,v} \log \hat{y}_{t,v}
$$

где $y_{t,v} = 1$ только для правильного токена, $\hat{y}_{t,v}$ — предсказанная вероятность.

### Связь с perplexity

$$
\text{PPL} = e^{\mathcal{L}} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(w_t \mid w_{<t})\right)
$$

| Модель | Perplexity |
|--------|-----------|
| GPT-2 | 18.3 |
| GPT-3 | 8.9 |
| LLaMA-2 70B | 3.3 |
| GPT-4 | ~2.5 |

Perplexity = среднее число вариантов, между которыми модель колеблется.

### Почему не MSE?

MSE между one-hot вектором и вероятностями — плохая идея:
- Не учитывает логарифмическую природу вероятностей
- Маленькие ошибки на редких токенах теряются
- Кросс-энтропия напрямую оптимизирует likelihood

---

## 3. Предобучение (Pre-training)

### Маскированный языковое моделирование (MLM)

Используется в BERT: случайно маскируем 15% токенов, предсказываем их.

$$
\mathcal{L}_{\text{MLM}} = -\mathbb{E}\left[\log P(w_{\text{mask}} \mid w_{\setminus \text{mask}})\right]
$$

### Causal языковое моделирование (CLM)

Используется в GPT: предсказываем следующий токен.

$$
\mathcal{L}_{\text{CLM}} = -\mathbb{E}\left[\log P(w_t \mid w_{<t})\right]
$$

### Масштабы предобучения

| Модель | Параметры | Данные (токены) | GPU-часы | Стоимость |
|--------|-----------|-----------------|----------|-----------|
| GPT-2 | 1.5B | 40B | ~500 | $50K |
| GPT-3 | 175B | 300B | ~35,000 | $4.6M |
| LLaMA-2 70B | 70B | 2T | ~170,000 | $20M |
| GPT-4 | ~1.8T | ~13T | ~$100M |

---

## 4. Fine-tuning

### SFT (Supervised Fine-Tuning)

Обучение на парах (instruction, response):

$$
\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim \mathcal{D}}\left[\log P(y \mid x; \theta)\right]
$$

Примеры данных:
```
Instruction: "Объясни квантовую запутанность простыми словами"
Response: "Представь два волшебных кубика..."
```

Типичные размеры: 10K–100K примеров, 1–5 эпох, lr в 10–100 раз меньше pretrain.

### PEFT (Parameter-Efficient Fine-Tuning)

Зачем замораживать 99% параметров:
- GPT-3 175B в fp32 = 700 GB — не влезает в GPU
- Полный fine-tune портит pretrain знания (catastrophic forgetting)

### LoRA (Low-Rank Adaptation)

К каждому весовому тензору добавляем низкоранговое приращение:

$$
W' = W + \Delta W = W + BA
$$

где $W \in \mathbb{R}^{d \times k}$, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$.

**Подсчёт параметров** для $W \in \mathbb{R}^{4096 \times 4096}$ при $r = 16$:
- Оригинал: 16.8M
- LoRA: $4096 \times 16 + 16 \times 4096 = 131K$ (в 128 раз меньше!)

Scaling factor $\alpha$:

$$
\Delta W = \frac{\alpha}{r} BA
$$

### QLoRA

Квантизация базы в 4-bit + LoRA адаптеры в bf16:
- Базовая модель: 70B × 0.5 bytes = 35 GB (влезает в одну A100)
- Адаптеры: ~50 MB
- Качество: 97–99% от полного fine-tune

---

## 5. RLHF (Reinforcement Learning from Human Feedback)

### Pipeline

```
SFT модель → Генерация ответов → Человеческие оценки → Reward Model → PPO оптимизация
```

### Шаг 1: Reward Model

На основе парных сравнений обучаем модель предсказывать предпочтение:

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}\left[\log \sigma\left(r(x, y_w) - r(x, y_l)\right)\right]
$$

где $y_w$ — предпочтительный ответ, $y_l$ — менее предпочтительный, $r$ — reward model, $\sigma$ — сигмоида.

### Шаг 2: PPO (Proximal Policy Optimization)

Оптимизируем языковую модель, используя reward model как среду:

$$
\text{objective} = \mathbb{E}\left[r(x, y)\right] - \beta \cdot \text{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right]
$$

KL-штраф не даёт модели уйти слишком далеко от SFT (предотвращает reward hacking).

**PPO clip**:

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \hat{A}, \text{clip}\left(\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}, 1 \pm \epsilon\right) \hat{A}\right)\right]
$$

### Проблемы RLHF

1. **Reward hacking** — модель эксплуатирует баги reward model
2. **Стоимость** — 4 модели одновременно (policy, ref, reward, critic)
3. **Нестабильность** — PPO чувствителен к гиперпараметрам
4. **Человеческие biases** — оценщики непоследовательны

---

## 6. DPO (Direct Preference Optimization)

### Идея

Обойти reward model — оптимизировать напрямую по парам предпочтений.

### Вывод

Из оптимальности KL-ограниченной задачи следует:

$$
r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Подставляем в loss Bradley-Terry модели:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

### Преимущества перед RLHF

| Критерий | RLHF | DPO |
|----------|------|-----|
| Моделей | 4 | 2 |
| Стабильность | Низкая | Высокая |
| Реализация | Сложная | Простой loss |
| Качество | Высокое | Сопоставимое |
| GPU память | 4× модель | 2× модель |

---

## 7. Scaling Laws

### Kaplan et al. (2020)

Loss монотонно зависит от трёх факторов:

$$
L(N, D, C) \approx \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_{\text{irreducible}}
$$

где:
- $N$ — число параметров
- $D$ — размер датасета (токены)
- $C \approx 6ND$ — суммарный FLOPs

Эмпирические показатели: $\alpha \approx 0.076$, $\beta \approx 0.095$.

### Chinchilla Scaling (Hoffmann et al., 2022)

**Оптимальное** соотношение: модель и датасет должны расти пропорционально.

$$
D_{\text{opt}} \approx 20 \times N
$$

| Модель | Параметры | Оптимальные токены | Реальные токены |
|--------|-----------|-------------------|-----------------|
| GPT-3 175B | 175B | 3.7T | 300B (недообучена) |
| Chinchilla 70B | 70B | 1.4T | 1.4T (оптимально) |
| LLaMA-2 70B | 70B | 1.4T | 2T (переобучена по токенам — ок) |

### Практические выводы

1. **Больше данных > больше параметров** — Chinchilla 70B побила Gopher 280B
2. **Ранний признак** — если val loss продолжает падать — модель мала для данных
3. **Инфраструктура** — обучение LLaMA-2 70B = 1,724,352 GPU-часов на A100

### Масштабирование и Emergent Abilities

Некоторые способности появляются скачкообразно при определённом масштабе:

- **In-context learning**: ~10B параметров
- **Chain-of-thought**: ~100B параметров
- **Сложная арифметика**: ~500B+ параметров

---

## 8. Инференс и квантизация

### Потребность в памяти

Размер модели в разных форматах:

$$
\text{Memory} = \frac{N \times \text{bytes per param}}{10^9} \text{ GB}
$$

| Формат | Байт/параметр | LLaMA-2 7B | LLaMA-2 70B |
|--------|---------------|------------|-------------|
| fp32 | 4 | 28 GB | 280 GB |
| fp16 | 2 | 14 GB | 140 GB |
| int8 | 1 | 7 GB | 70 GB |
| int4 (GPTQ) | 0.5 | 3.5 GB | 35 GB |

### KV Cache

На каждом слое хранятся ключи и значения для всех предыдущих токенов:

$$
\text{KV Cache} = 2 \times n_{\text{layers}} \times d_{\text{model}} \times n_{\text{heads}} \times \text{seq-len} \times \text{bytes}
$$

Для LLaMA-2 70B при seq_len = 4096, fp16:

$$
2 \times 80 \times 8192 \times 64 \times 4096 \times 2 \approx 160 \text{ GB}
$$

Поэтому длинный контекст дорогой!

### Методы квантизации

**Post-training quantization (PTQ)**: GPTQ, AWQ — обучаемая калибровка на маленьком датасете.

**Quantization-aware training (QAT)**: квантизация встроена в обучение.

Качество: int4 теряет 1–3% на бенчмарках, int8 — менее 1%.

---

## 9. Современные архитектуры

### Mixture of Experts (MoE)

Вместо одной FFN — несколько экспертов, маршрутизация через gate:

$$
\text{MoE}(x) = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)
$$

где $g_i(x) = \text{softmax}(\text{top-k}(W_g \cdot x))$ — gate выбирает top-k экспертов.

| Модель | Всего параметров | Активных | Экспертов | Top-k |
|--------|-----------------|----------|-----------|-------|
| Mixtral 8×7B | 46.7B | 12.9B | 8 | 2 |
| DeepSeek-V3 | 671B | 37B | 256 | 8 |

Активных параметров в 3–18 раз меньше — быстрее инференс.

### Grouped Query Attention (GQA)

 KV heads < Q heads — делим KV между группами запросов:

| Модель | Q heads | KV heads | Ratio |
|--------|---------|----------|-------|
| LLaMA-2 70B | 64 | 8 | 8:1 |
| LLaMA-3 70B | 64 | 8 | 8:1 |
| Mistral 7B | 32 | 8 | 4:1 |

KV cache уменьшается пропорционально.

---

## 10. Практика

### Полный цикл обучения маленькой GPT

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- Токенизация ---
text = "Вставьте большой текст..."
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
vocab_size = len(chars)

def encode(s): return [stoi[c] for c in s]
data = torch.tensor(encode(text), dtype=torch.long)

# --- Датасет ---
block_size = 256

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

# --- Модель (из лекции 11) ---
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        
        # Подсчёт параметров
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Параметров: {n_params / 1e6:.2f}M")
    
    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# --- Обучение ---
model = MiniGPT(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

dataset = CharDataset(data, block_size)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for epoch in range(50):
    total_loss = 0
    for x, y in loader:
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), 
            y.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    avg = total_loss / len(loader)
    ppl = torch.exp(torch.tensor(avg))
    print(f"Epoch {epoch}: loss={avg:.4f}, ppl={ppl:.1f}")
```

### DPO fine-tuning

```python
def dpo_loss(policy_model, ref_model, x, y_w, y_l, beta=0.1):
    """
    y_w — предпочтительный ответ
    y_l — менее предпочтительный ответ
    """
    # Лог-вероятности под policy и reference
    pi_w = policy_model.log_prob(y_w, x)
    pi_l = policy_model.log_prob(y_l, x)
    ref_w = ref_model.log_prob(y_w, x)
    ref_l = ref_model.log_prob(y_l, x)
    
    # Логит разности
    log_ratio_w = pi_w - ref_w  # log π(y_w|x) / π_ref(y_w|x)
    log_ratio_l = pi_l - ref_l  # log π(y_l|x) / π_ref(y_l|x)
    
    # DPO loss
    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l))
    return loss.mean()

# Использование
ref_model = copy.deepcopy(sft_model)  # Замороженный SFT
for param in ref_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-6)

for batch in dpo_dataloader:
    loss = dpo_loss(policy_model, ref_model, 
                    batch['prompt'], batch['chosen'], batch['rejected'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Подсчёт FLOPs обучения

```python
def estimate_training_flops(n_params, n_tokens, forward_flops=6, backward_flops=12):
    """
    forward: ~2 * n_params FLOPs per token
    backward: ~4 * n_params FLOPs per token
    total per token: ~6 * n_params
    """
    total = n_params * n_tokens * (forward_flops + backward_flops)
    print(f"FLOPs: {total:.2e}")
    print(f"GPU-часов (A100=312 TFLOPS): {total / 312e12 / 3600:.0f}")

# LLaMA-2 7B
estimate_training_flops(7e9, 2e12)
# FLOPs: 2.52e+23
# GPU-часов: ~224,000

# LLaMA-2 70B
estimate_training_flops(70e9, 2e12)
# FLOPs: 2.52e+24
# GPU-часов: ~2,243,000
```

---

## Итоги курса

### Карта взаимосвязей

```
Перцептрон (Л1) → MLP (Л2) → Обучение: GD (Л5) → Backprop (Л6)
                      ↓                                    ↓
              Векторы (Л3) → Эмбеддинги (Л4)        CNN (Л7-8)
                                                       ↓
                              GAN (Л9) ← ─ ─ ─ ─ ─ ─ ┘
                                ↓
                           Diffusion/VAE (Л10)
                                ↓
                           Трансформер (Л11)
                                ↓
                           LLM (Л12)
```

### Что вы теперь умеете

1. Понимать математику за каждой операцией в нейросети
2. Реализовывать компоненты с нуля на NumPy
3. Читать и понимать статьи (Attention Is All You Need, DDPM, DPO)
4. Оценивать стоимость обучения и инференса
5. Выбирать архитектуру под задачу

### Что дальше

- **Распределённое обучение** — FSDP, ZeRO, tensor/pipeline parallelism
- **Multimodal** — CLIP, LLaVA, GPT-4V
- **Agent systems** — tool use, chain-of-thought, reasoning
- **On-device** — дистилляция, прунинг, мобильный инференс
