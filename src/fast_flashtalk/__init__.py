__all__ = ["FlashTalkPipeline", "Image", "Audio"]


def __getattr__(name: str):
    if name == "FlashTalkPipeline":
        from .pipeline import FlashTalkPipeline

        return FlashTalkPipeline
    if name == "Image":
        from osc_data.image import Image

        return Image
    if name == "Audio":
        from osc_data.audio import Audio

        return Audio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
