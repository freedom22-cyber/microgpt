# MicroGPT

A minimal, dependency-free implementation of a GPT-2 style transformer model in pure Python. This project demonstrates the core algorithms of modern large language models with the simplest possible code.

## 📋 概述

MicroGPT 是基于 Andrej Karpathy 的原始概念实现的极简 GPT 模型。该项目的目标是用最少的代码展示训练和推理一个 GPT 所需的全部算法，没有任何外部依赖。

- **纯 Python**: 无任何 NumPy、PyTorch 等依赖
- **完整算法**: 包括 Tokenization、Autograd、Transformer 架构、训练和推理
- **可读性**: 代码注释清晰，易于学习理解
- **教学导向**: 理想的学习 GPT 内部工作原理的工具

## 🚀 快速开始

### 前置要求

- Python 3.7+
- 无其他依赖

### 使用方法

```bash
python microgpt.py
```

程序会自动：
1. 下载示例数据集（常见名字列表）
2. 初始化模型参数
3. 训练模型
4. 生成新的示例

## 📚 核心组件

### 1. **Tokenizer**
- 将文本字符串转换为整数序列（tokens）
- 基于唯一字符构建词汇表
- 支持特殊的序列开始 (BOS) token

### 2. **自动微分 (Autograd)**
- 实现完整的反向传播算法
- 通过计算图进行链式法则计算
- `Value` 类跟踪前向和后向过程

### 3. **Transformer 架构**
- 单层 Transformer 模型
- 多头注意力机制
- 前馈神经网络 (MLP)
- RMSNorm 层正则化
- ReLU 激活函数

### 4. **模型参数**

默认配置：
```python
n_layer = 1          # Transformer 层数
n_embd = 16          # 嵌入维度
block_size = 16      # 最大上下文长度
n_head = 4           # 注意力头数
```

## 🔧 主要特性

- **Token 嵌入**: 学习每个字符的向量表示
- **位置编码**: 编码序列中的位置信息
- **多头注意力**: 并行学习不同的表示子空间
- **前馈网络**: 非线性变换层
- **损失函数**: 交叉熵损失用于下一个 token 预测

## 📖 使用示例

```python
# 模型自动：
# 1. 加载数据集
# 2. 初始化所有参数
# 3. 在数据上进行若干步训练
# 4. 生成新的示例序列
```

## 🎯 学习路径

1. **理解 Tokenization**: 查看如何将字符串转换为 tokens
2. **学习 Autograd**: 研究 `Value` 类如何计算梯度
3. **研究 Transformer**: 理解注意力机制和前馈网络
4. **追踪训练**: 观察损失函数如何变化
5. **生成文本**: 使用训练的模型生成新序列

## 💡 项目特点

- ✅ 完全从零实现，无第三方库
- ✅ 清晰的代码结构和注释
- ✅ 演示所有核心概念
- ✅ 适合教学和学习
- ✅ 可扩展性，易于修改参数

## 🔍 技术细节

### 前向传播
- Token 嵌入 + 位置嵌入
- 多头注意力计算
- 前馈网络变换
- Logits 输出和 Softmax 概率

### 后向传播
- 通过计算图的自动微分
- 链式法则应用
- 参数梯度计算

### 优化
- 简单的随机梯度下降 (SGD)
- 可配置的学习率

## 📝 文件说明

- `microgpt.py` - 完整的模型实现
- `input.txt` - 输入数据文件（首次运行时自动下载）
- `README.md` - 本文档

## 🤝 贡献

欢迎提出改进建议、bug 报告或代码优化！

## 📄 许可证

基于原始项目的使用方式。

## 🙏 致谢

灵感来源：[Andrej Karpathy](https://github.com/karpathy) 的 makemore 项目

---

**注意**: 这是一个教学项目，用于理解 GPT 的基本原理。对于生产环境，请使用 PyTorch、TensorFlow 或其他优化的框架。
