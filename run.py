from fast_flashtalk.pipeline import FlashTalkPipeline, Image, Audio


ckpt_dir = "checkpoints/Soul-AILab/SoulX-FlashTalk-14B"
wav2vec_dir = "checkpoints/TencentGameMate/chinese-wav2vec2-base"
pipeline = FlashTalkPipeline(
    checkpoint_dir=ckpt_dir,
    wav2vec_dir=wav2vec_dir,
    num_persistent_param_in_dit=15_000_000_000,
)

image = Image(uri="examples/man.png")
audio = Audio(uri="examples/man.mp3")
result = pipeline.generate(
    input_prompt="男子穿白衬衫，面向镜头。室内办公室背景，绿色植物点缀，墙上挂有城市地图，光线明亮柔和。近景，平视视角，标准镜头拍摄。写实风格。",
    audio=audio,
    image=image,
)
