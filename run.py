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
    input_prompt="A person is talking. Only the foreground characters are moving, the background remains static.",
    audio=audio,
    image=image,
)
