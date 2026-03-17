# Fast-FlashTalk

基于 [FlashTalk](https://github.com/Soul-AI-Lab/FlashTalk) 的高性能推理优化版本，专为 RTX 4090 显卡优化，在保持生成质量的同时显著降低显存占用并提升推理速度，实测可达 **3 倍加速**。

## 优化项

### 1. DiT 动态参数加载

实现了 DiT 模型参数在 CPU/GPU 之间的动态调度。通过 `num_persistent_param_in_dit` 参数控制常驻 GPU 的参数量，超出部分自动 offload 到 CPU，推理时按需加载，从而在有限显存下运行 14B 参数的 DiT 模型。

### 2. GemLite A8W8 量化

使用 GemLite 对 DiT 模型进行 A8W8 int8 动态量化，将线性层的权重和激活值量化为 int8，大幅降低显存占用和计算开销。部分关键模块（time_embedding、head 等）被排除在量化之外以保证精度。

### 3. apply_rope 算子优化

使用 flash_attention 提供的 `apply_rotary_emb` 替代原始的逐元素复数运算实现 RoPE，利用 CUDA kernel 融合加速旋转位置编码的计算。

### 4. SageAttention

集成 [SageAttention](https://github.com/thu-ml/SageAttention) 替代标准注意力计算，支持 `sageattn` 和 `sageattn_varlen` 两种模式。对短序列（< 512）自动回退至 flash_attn 以保持最优性能。

### 5. T5 Cache

对 T5 编码器的推理结果使用 `lru_cache` 进行缓存（默认 maxsize=20），相同文本 prompt 的重复推理直接返回缓存结果，避免重复计算。

## 安装

```bash
pip install -r requirements.txt
```

## 快速使用

```python
from flash_talk import FlashTalkPipeline
from osc_data.image import Image
from osc_data.audio import Audio

pipeline = FlashTalkPipeline(
    checkpoint_dir="path/to/SoulX-FlashTalk-14B",
    wav2vec_dir="path/to/chinese-wav2vec2-base",
    num_persistent_param_in_dit=15_000_000_000,  # 常驻 GPU 的参数量，根据显存调整
)

image = Image(uri="path/to/image.jpeg")
audio = Audio(uri="path/to/audio.wav")

video = pipeline.generate(
    input_prompt="人物描述和场景描述",
    audio=audio,
    image=image,
    force_9_16=True,  # 强制 9:16 宽高比
)
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint_dir` | str | - | FlashTalk 模型权重目录 |
| `wav2vec_dir` | str | - | Wav2Vec2 模型目录 |
| `num_persistent_param_in_dit` | int | 15B | 常驻 GPU 的 DiT 参数量，显存不足时调小 |
