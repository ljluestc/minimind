# feat(attention): 在 MiniMind 中实现门控注意力机制

关闭 #702

## 问题

MiniMind 缺少内置的门控注意力机制来稳定训练过程中的注意力输出。问题 #702 要求：
- 在注意力输出中添加可学习的门控参数
- 支持 `headwise` 和 `elementwise` 两种门控模式（默认为 `elementwise`）
- 与 Flash Attention 和 KV cache 完全兼容
- 通过训练标志（如 `--attn_gate 1`）轻松启用

## 解决方案

在 `model/model_minimind.py` 中实现了门控注意力机制，并通过训练和推理入口点进行了连接。

### 1) 配置支持

添加了配置字段：
- `attn_gate` (bool) - 启用门控注意力
- `attn_gate_mode` (`elementwise` 或 `headwise`) - 门控模式

同时保持与现有别名 `attn_gate_type` 的兼容性。

### 2) 注意力模块门控

在 `Attention` 类中：
- 添加了可学习的门控参数 `attn_gate_param`
- 形状取决于模式：
  - `elementwise`: `(num_heads, head_dim)` - 逐元素门控
  - `headwise`: `(num_heads,)` - 按头门控
- 在注意力输出计算后应用门控（支持 flash 和非 flash 路径）：
  - `gate = 2 * sigmoid(attn_gate_param)`
  - `output = output * gate`

### 3) CLI/训练器标志连接

在主要训练脚本中添加并连接了训练器标志：
- `--attn_gate` - 启用门控注意力（0或1）
- `--attn_gate_type elementwise|headwise` - 指定门控模式

这些标志被传递给 `MiniMindConfig`。

涵盖的训练脚本：
- `trainer/train_pretrain.py`
- `trainer/train_full_sft.py`
- `trainer/train_lora.py`
- `trainer/train_dpo.py`
- `trainer/train_distillation.py`
- `trainer/train_grpo.py`
- `trainer/train_ppo.py`
- `trainer/train_agent.py`

### 4) 推理和文档支持

在推理脚本中暴露门控注意力选项：
- `eval_llm.py`
- `scripts/eval_toolcall.py`
- `scripts/serve_openai_api.py`

在 README 文件中记录使用方法。

## 文件变更

### 核心实现
- `model/model_minimind.py` - 门控注意力机制实现

### 训练脚本（添加标志支持）
- `trainer/train_pretrain.py`
- `trainer/train_full_sft.py`
- `trainer/train_lora.py`
- `trainer/train_dpo.py`
- `trainer/train_distillation.py`
- `trainer/train_grpo.py`
- `trainer/train_ppo.py`
- `trainer/train_agent.py`

### 推理脚本
- `eval_llm.py`
- `scripts/eval_toolcall.py`
- `scripts/serve_openai_api.py`

### 文档和测试
- `README.md` - 使用说明
- `README_en.md` - 英文使用说明
- `tests/test_gated_attention.py` - 核心门控注意力测试
- `tests/test_attention_gate.py` - 注意力门控集成测试

## 变更类型

- [ ] Bug 修复（不破坏现有功能的非破坏性修复）
- [x] 新功能（添加功能的非破坏性修复）
- [ ] 破坏性变更（可能导致现有功能无法正常工作的修复或功能）

## 如何测试？

已添加单元测试并本地验证通过：

**运行命令：**
```bash
python -m pytest tests/ -v
```

**测试结果：**
- 运行 9 个测试
- 所有测试通过 ✓

**测试覆盖内容：**
- Elementwise 门控参数形状验证
- Headwise 门控参数形状验证
- 禁用门控时的默认行为
- KV cache 路径与门控兼容性
- Flash Attention 路径与门控兼容性
- 门控参数对输出幅度的控制

## 风险评估

**低风险：**
- 该功能是可选的，通过 `--attn_gate` 标志启用（默认禁用）
- 默认行为保持不变，不影响现有模型
- 关键路径均由针对性测试覆盖
- 与现有 Flash Attention 和 KV cache 机制完全兼容

## 回滚计划

如果发生回归问题：
1. 还原 `model/model_minimind.py` 中的门控注意力添加
2. 还原训练/推理脚本中的 `attn_gate` 和 `attn_gate_type` 标志连接
3. 还原此 PR 中的相关文档和测试

## 使用示例

### 启用 Elementwise 门控（推荐）

```bash
python trainer/train_pretrain.py \
  --attn_gate 1 \
  --attn_gate_type elementwise \
  # ... 其他参数
```

### 启用 Headwise 门控

```bash
python trainer/train_pretrain.py \
  --attn_gate 1 \
  --attn_gate_type headwise \
  # ... 其他参数
```

### 推理时启用

```bash
python eval_llm.py \
  --attn_gate 1 \
  --attn_gate_type elementwise \
  # ... 其他参数
```

## 性能影响

门控注意力引入的计算开销最小：
- **参数增加：** 
  - Elementwise：每个头 `head_dim` 个额外参数
  - Headwise：每个头 1 个额外参数
- **计算开销：** 每个注意力头增加一个乘法和 sigmoid 操作

对于 12 亿参数的模型，总参数增加 < 0.01%。

## 验证清单

- [x] 门控注意力机制已在 `model/model_minimind.py` 中实现
- [x] 支持 elementwise 和 headwise 两种模式
- [x] 与 Flash Attention 兼容
- [x] 与 KV cache 兼容
- [x] 所有训练脚本中已添加标志支持
- [x] 所有推理脚本中已添加标志支持
- [x] 添加了 9 个单元测试，全部通过
- [x] 文档已更新
- [x] 代码已推送到私有分支 `private/issue-702-gated-attention`
