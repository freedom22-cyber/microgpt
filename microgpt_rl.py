"""
MicroGPT + Reinforcement Learning (REINFORCE Policy Gradient)

Extends the base MicroGPT with RL fine-tuning using policy gradients.
After supervised pre-training on a names dataset, the model is fine-tuned
with REINFORCE to steer generation toward a target property — demonstrating
the core algorithm behind RLHF (RL from Human Feedback) used to align LLMs.

Pipeline:
  Phase 1 — Supervised pre-training (next-token prediction)
  Phase 2 — Pre-RL evaluation baseline
  Phase 3 — RL fine-tuning with REINFORCE + moving-average baseline
  Phase 4 — Post-RL evaluation & comparison
"""

import os
import math
import random
import sys
sys.setrecursionlimit(25000)
random.seed(42)

# ─── Dataset ───────────────────────────────────────────────
if not os.path.exists('input.txt'):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ─── Tokenizer ─────────────────────────────────────────────
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ─── Autograd ──────────────────────────────────────────────
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo, visited = [], set()
        def _build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    _build(c)
                topo.append(v)
        _build(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

# ─── Model Parameters ─────────────────────────────────────
n_layer, n_embd, block_size, n_head = 1, 16, 16, 4
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
sd = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    for suffix in ('attn_wq', 'attn_wk', 'attn_wv', 'attn_wo'):
        sd[f'layer{i}.{suffix}'] = matrix(n_embd, n_embd)
    sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in sd.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ─── Model Architecture ───────────────────────────────────
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    s = sum(exps)
    return [e / s for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    x = [t + p for t, p in zip(sd['wte'][token_id], sd['wpe'][pos_id])]
    x = rmsnorm(x)
    for li in range(n_layer):
        xr = x
        x = rmsnorm(x)
        q = linear(x, sd[f'layer{li}.attn_wq'])
        k = linear(x, sd[f'layer{li}.attn_wk'])
        v = linear(x, sd[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        xa = []
        for h in range(n_head):
            hs = h * head_dim
            qh = q[hs:hs+head_dim]
            kh = [ki[hs:hs+head_dim] for ki in keys[li]]
            vh = [vi[hs:hs+head_dim] for vi in values[li]]
            al = [sum(qh[j] * kh[t][j] for j in range(head_dim)) / head_dim**0.5
                  for t in range(len(kh))]
            aw = softmax(al)
            xa.extend([sum(aw[t] * vh[t][j] for t in range(len(vh)))
                       for j in range(head_dim)])
        x = linear(xa, sd[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, xr)]
        xr = x
        x = rmsnorm(x)
        x = [xi.relu() for xi in linear(x, sd[f'layer{li}.mlp_fc1'])]
        x = linear(x, sd[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, xr)]
    return linear(x, sd['lm_head'])

# ─── Adam Optimizer ────────────────────────────────────────
lr0, beta1, beta2, eps = 0.01, 0.85, 0.99, 1e-8
m_buf = [0.0] * len(params)
v_buf = [0.0] * len(params)
step_count = 0

def adam_step(lr):
    global step_count
    step_count += 1
    for i, p in enumerate(params):
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad ** 2
        mh = m_buf[i] / (1 - beta1 ** step_count)
        vh = v_buf[i] / (1 - beta2 ** step_count)
        p.data -= lr * mh / (vh ** 0.5 + eps)
        p.grad = 0

# ─── Helper: generate names ───────────────────────────────
def generate_names(n_samples, temperature=0.5):
    names = []
    for _ in range(n_samples):
        kc, vc = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        tid, chars = BOS, []
        for pos in range(block_size):
            probs = softmax([l / temperature for l in gpt(tid, pos, kc, vc)])
            tid = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if tid == BOS:
                break
            chars.append(uchars[tid])
        names.append(''.join(chars))
    return names

# ═══════════════════════════════════════════════════════════
# Phase 1: Supervised Pre-training
# ═══════════════════════════════════════════════════════════
num_pretrain = 200
print(f"\n{'='*55}")
print(f" Phase 1: Supervised Pre-training ({num_pretrain} steps)")
print(f"{'='*55}")

for step in range(num_pretrain):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    kc, vc = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos in range(n):
        probs = softmax(gpt(tokens[pos], pos, kc, vc))
        losses.append(-probs[tokens[pos + 1]].log())
    loss = (1 / n) * sum(losses)
    loss.backward()
    adam_step(lr0 * (1 - step / num_pretrain))
    if (step + 1) % 50 == 0 or step == 0:
        print(f"  step {step+1:4d}/{num_pretrain} | loss {loss.data:.4f}")

# ═══════════════════════════════════════════════════════════
# Phase 2: Pre-RL Evaluation
# ═══════════════════════════════════════════════════════════
TARGET = 'a'
print(f"\n{'='*55}")
print(f" Phase 2: Pre-RL Evaluation (target: starts with '{TARGET}')")
print(f"{'='*55}")

pre_names = generate_names(30)
for i, nm in enumerate(pre_names):
    tag = " <--" if nm and nm[0] == TARGET else ""
    print(f"  {i+1:2d}. {nm}{tag}")
pre_hit = sum(1 for nm in pre_names if nm and nm[0] == TARGET)
print(f"\n  Hit rate: {pre_hit}/30 ({100*pre_hit/30:.0f}%)")

# ═══════════════════════════════════════════════════════════
# Phase 3: RL Fine-tuning (REINFORCE)
# ═══════════════════════════════════════════════════════════
num_rl = 80
rl_lr = 0.002
rl_temp = 0.8

print(f"\n{'='*55}")
print(f" Phase 3: REINFORCE RL Fine-tuning ({num_rl} steps)")
print(f"  Reward: +2 if name starts with '{TARGET}', -0.5 otherwise")
print(f"  Bonus:  +0.5 if length in [3,7], -0.5 otherwise")
print(f"{'='*55}")

def reward_fn(name):
    if not name:
        return -2.0
    r = 2.0 if name[0] == TARGET else -0.5
    r += 0.5 if 3 <= len(name) <= 7 else -0.5
    return r

baseline = 0.0
reward_log = []

for step in range(num_rl):
    # 1) Roll out a trajectory by sampling from the policy
    kc, vc = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    tid, log_probs, gen = BOS, [], []
    for pos in range(block_size):
        probs = softmax([l / rl_temp for l in gpt(tid, pos, kc, vc)])
        tid = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        log_probs.append(probs[tid].log())
        if tid == BOS:
            break
        gen.append(tid)

    name = ''.join(uchars[t] for t in gen)
    reward = reward_fn(name)
    reward_log.append(reward)

    # 2) Update baseline (moving average for variance reduction)
    baseline = 0.9 * baseline + 0.1 * reward
    advantage = reward - baseline

    # 3) Policy gradient: L = -Σ log π(aₜ|sₜ) · (R - b)
    if log_probs:
        pg_loss = sum(-lp * advantage for lp in log_probs) / len(log_probs)
        pg_loss.backward()
        adam_step(rl_lr)

    if (step + 1) % 10 == 0 or step == 0:
        avg_r = sum(reward_log[-10:]) / min(10, len(reward_log))
        print(f"  step {step+1:3d}/{num_rl} | R {reward:+.1f} | avg_R {avg_r:+.2f} | {name}")

# ═══════════════════════════════════════════════════════════
# Phase 4: Post-RL Evaluation
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f" Phase 4: Post-RL Evaluation")
print(f"{'='*55}")

post_names = generate_names(30)
for i, nm in enumerate(post_names):
    tag = " <--" if nm and nm[0] == TARGET else ""
    print(f"  {i+1:2d}. {nm}{tag}")
post_hit = sum(1 for nm in post_names if nm and nm[0] == TARGET)
print(f"\n  Hit rate: {post_hit}/30 ({100*post_hit/30:.0f}%)")

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print(f" RL Training Summary")
print(f"{'='*55}")
print(f"  Objective       : generate names starting with '{TARGET}'")
print(f"  Algorithm       : REINFORCE with moving-average baseline")
print(f"  Pre-RL hit rate : {pre_hit}/30 ({100*pre_hit/30:.0f}%)")
print(f"  Post-RL hit rate: {post_hit}/30 ({100*post_hit/30:.0f}%)")
delta = post_hit - pre_hit
if delta > 0:
    print(f"  Improvement     : +{delta} names (+{100*delta/30:.0f}% absolute)")
elif delta < 0:
    print(f"  Regression      : {delta} names ({100*delta/30:.0f}% absolute)")
else:
    print(f"  Change          : none")
print(f"  Avg RL reward   : {sum(reward_log)/len(reward_log):+.2f}")
