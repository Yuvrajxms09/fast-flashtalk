# Fast-FlashTalk

基于 [FlashTalk](https://github.com/Soul-AI-Lab/FlashTalk) 的高性能推理优化版本，专为 RTX 4090 显卡优化，在保持生成质量的同时显著降低显存占用并提升推理速度，实测可达 **2 倍加速**。

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

需要 **Python 3.11+**、支持 CUDA 的 NVIDIA GPU（推荐显存 24GB 及以上以运行 14B 模型），以及 **ffmpeg**。
```bash
# Debian/Ubuntu
sudo apt update && sudo apt install -y ffmpeg
# macOS（Homebrew
brew install ffmpeg
# conda
conda install -c conda-forge ffmpeg
```

```bash
pip install fast-flashtalk
```


## 模型与数据准备

推理前需自行下载模型权重：

```
modelscope download Soul-AILab/SoulX-FlashTalk-14B --local_dir checkpoints/Soul-AILab/SoulX-FlashTalk-14B
modelscope download TencentGameMate/chinese-wav2vec2-base --local_dir checkpoints/TencentGameMate/chinese-wav2vec2-basechinese-wav2vec2-base
```

## 使用说明

首次创建 `FlashTalkPipeline` 时会从磁盘加载多路权重、完成量化与显存调度初始化；首次调用 `generate` 时还可能包含 CUDA 预热、部分算子首次执行等一次性开销，因此**第一次运行整体会明显慢于后续同进程内的推理**，属正常现象。同一进程内再次生成通常会快很多。

### 最小示例

```python
from fast_flashtalk import Audio, FlashTalkPipeline, Image

checkpoint_dir = "path/to/SoulX-FlashTalk-14B"
wav2vec_dir = "path/to/chinese-wav2vec2-base"

pipeline = FlashTalkPipeline(
    checkpoint_dir=checkpoint_dir,
    wav2vec_dir=wav2vec_dir,
    num_persistent_param_in_dit=15_000_000_000,
)

image = Image(uri="path/to/portrait.png")
audio = Audio(uri="path/to/speech.wav")

video = pipeline.generate(
    input_prompt="人物与场景描述，用于引导画面风格与内容。",
    audio=audio,
    image=image,
)
# 添加音频与视频合并
#video.merge_audio(audio)
video.save("test.mp4")
```

`Image`、`Audio` 使用 `uri` 指向本地图片或音频文件；音频会按管线内配置的采样率重采样。

### `FlashTalkPipeline` 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint_dir` | `str` | 必填 | FlashTalk 模型权重根目录 |
| `wav2vec_dir` | `str` | 必填 | Wav2Vec2 模型本地目录 |
| `num_persistent_param_in_dit` | `int` | `10_000_000_000` | 常驻 GPU 的 DiT 参数个数上限，显存紧张时适当调小 |

### `generate` 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_prompt` | `str` | 必填 | 文本提示，描述人物、场景、镜头等 |
| `audio` | `Audio` | 必填 | 驱动口型与节奏的音频 |
| `image` | `Image` | 必填 | 条件图像（人物/画面参考） |
| `audio_encode_mode` | `"stream"` \| `"once"` | `"once"` | 音频编码方式：`once` 整段编码后按块切分；`stream` 按流式块编码，更省内存 |

返回值类型为 `osc_data.video.Video`，可通过 `.data` 等属性访问帧数据（具体以 `osc_data` 文档为准）。

## 许可证与致谢

上游实现与权重归属请参考 [FlashTalk](https://github.com/Soul-AI-Lab/FlashTalk) 及相应模型许可。
