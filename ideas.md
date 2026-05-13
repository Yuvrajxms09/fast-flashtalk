# Visual Consistency Ideas

Ranked from most straightforward to more invasive.

## 1. Existing knobs to sweep first

These are already exposed in `FlashTalkPipeline.generate()` and can be tested without changing the architecture:

- `frame_num`
- `motion_frames_num`
- `sampling_steps`
- `color_correction_strength`

What they do:

- `frame_num`: larger chunk window, more context per chunk, fewer chunk boundaries
- `motion_frames_num`: more overlap, smoother transitions, but more repeated work
- `sampling_steps`: quality vs speed, not a core consistency fix
- `color_correction_strength`: helps color matching across chunks, but only lightly

## 2. Temporal crossfade at chunk boundaries

The code already has a single stitch point where overlap is dropped.
The VAE also already uses overlap blending ideas through `blend_v()` and `blend_h()`.

The same overlap-and-blend concept can be ported from spatial seams to temporal seams:

- keep the overlap frames from the previous chunk
- blend them with the first frames of the next chunk
- append only the blended transition plus new non-overlap frames

This is the easiest real improvement beyond knob tuning.

## 3. Carry more latent state across chunks

This is the main architectural weakness today.
The current pipeline preserves only a small tail state, not a richer long-term memory.

A stronger fix would be to:

- preserve more of the previous chunk latent
- re-inject it into the next chunk
- reduce the amount of fresh noise at each chunk boundary

Implemented in `fast-flashtalk` as the `latent_carryover_steps` flag, which carries a tail of the previous chunk's denoised latent forward into the next chunk.

This should help the most with long-run drift, but it is more invasive than blending.

## 4. Periodic re-anchoring

Every N chunks, re-apply the source image or a stable keyframe more strongly.

This can help keep:

- identity
- background
- lighting

more stable over long videos.

## 5. Drift detection

If a chunk starts drifting too far from the reference, adjust the next chunk by:

- increasing overlap
- strengthening blending
- re-anchoring to the source frame

This is more control logic than model logic, so it is better as a later refinement.

## 6. FluxRT-style adaptive cache refresh

This is the most useful idea borrowed from `FluxRT`.

Where it appears in `FluxRT`:

- [`/Users/yuvraj/Desktop/Italk/FluxRT/src/fluxrt/stream_processor/update_controller.py`](\/Users/yuvraj/Desktop/Italk/FluxRT/src/fluxrt/stream_processor/update_controller.py)
- [`/Users/yuvraj/Desktop/Italk/FluxRT/src/fluxrt/stream_processor/transformer_flux2.py`](\/Users/yuvraj/Desktop/Italk/FluxRT/src/fluxrt/stream_processor/transformer_flux2.py)

What `FluxRT` does:

- compares the current frame against a cached frame
- creates a mask / update decision based on how much changed
- lets the model skip, execute-only, or execute-and-update
- keeps a reference image / cache state that can be reset when needed

Why this helps `fast-flashtalk`:

- the pipeline already has chunk boundaries and a cached tail state
- we can measure drift between chunks instead of always treating them the same
- if drift is small, continue with normal overlap
- if drift is large, re-anchor harder, increase blending, or refresh the cached latent state

This is a better long-video control strategy than blindly using the same overlap for every chunk.

## What not to prioritize

- `audio_encode_mode="stream"` for visual consistency
- `cached_audio_duration` for visual consistency
- `sampling_steps` beyond the current max-quality setting
- `color_correction_strength` as the main fix

These affect memory, audio handling, or color matching more than actual temporal consistency.

## Recommended implementation order

1. Sweep `frame_num` and `motion_frames_num`
2. Add temporal crossfade at chunk boundaries
3. Add FluxRT-style adaptive drift detection / cache refresh
4. If needed, carry richer latent state across chunks
5. Add periodic re-anchoring only if long-run drift remains
