"""Small synthetic API graphs for exercising scene wiring without model inference."""

import json


def sample_graph(version="2.5", *, legacy_decoder=False):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}

    graph = {
        "model": node("UNETLoader", unet_name="test-transformer.safetensors", weight_dtype="default"),
        "clip": node("CLIPLoader", clip_name="test-text-encoder.safetensors", type="ltxv", device="default"),
        "video_vae": node("VAELoader", vae_name="test-video-vae.safetensors"),
        "audio_vae": node("VAELoader", vae_name="test-audio-vae.safetensors"),
        "first_seed": node("RandomNoise", noise_seed=7),
        "sampler_choice": node("KSamplerSelect", sampler_name="euler"),
        "schedule": node("ManualSigmas", sigmas="1.0, 0.25, 0.0"),
        "video": node("CreateVideo", images=["decode", 0], audio=["decode_audio", 0], fps=24),
        "save": node("SaveVideo", video=["video", 0], filename_prefix="fixture-scene", format="auto", codec="auto"),
    }
    if version == "h3":
        graph.update({
            "latent": node("MiniMaxH3ImageToVideo", prompt="", width=320, height=320, length=73, clip=["clip", 0]),
            "guidance": node("BasicGuider", model=["model", 0], conditioning=["latent", 0]),
            "first_sample": node("SamplerCustomAdvanced", noise=["first_seed", 0], guider=["guidance", 0],
                                 sampler=["sampler_choice", 0], sigmas=["schedule", 0], latent_image=["latent", 1]),
            "decode": node("VAEDecode", samples=["first_sample", 0], vae=["video_vae", 0]),
            "decode_audio": node("VAEDecodeAudio", samples=["first_sample", 0], vae=["audio_vae", 0]),
        })
        return graph

    negative = "text"
    model = "model"
    if version == "2.3":
        negative = "negative"
        model = "lora"
        graph["model"] = node("CheckpointLoaderSimple", ckpt_name="test-checkpoint.safetensors")
        graph[negative] = node("CLIPTextEncode", clip=["clip", 0], text="test negative")
        graph[model] = node("LoraLoaderModelOnly", model=["model", 0], lora_name="test-distill.safetensors",
                            strength_model=0.4)
    upscaler = {"2.3": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                "2.5": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"}[version]
    guidance = (node("CFGGuider", model=[model, 0], positive=["conditioning", 0], negative=["conditioning", 1], cfg=2)
                if version == "2.3" else
                node("LTXVDualCFGGuider", model=[model, 0], positive=["conditioning", 0], negative=["conditioning", 1],
                     video_cfg=1.0, audio_cfg=1.0))
    graph.update({
        "text": node("CLIPTextEncode", clip=["clip", 0], text=""),
        "conditioning": node("LTXVConditioning", positive=["text", 0], negative=[negative, 0], frame_rate=24),
        "guidance": guidance,
        "frame_count": node("PrimitiveInt", value=113),
        "latent": node("EmptyLTXVLatentVideo", width=320, height=320, length=["frame_count", 0], batch_size=1),
        "audio_latent": node("LTXVEmptyLatentAudio", frames_number=["frame_count", 0], frame_rate=24, batch_size=1,
                             audio_vae=["audio_vae", 0]),
        "first_joint": node("LTXVConcatAVLatent", video_latent=["latent", 0], audio_latent=["audio_latent", 0]),
        "first_sample": node("SamplerCustomAdvanced", noise=["first_seed", 0], guider=["guidance", 0],
                             sampler=["sampler_choice", 0], sigmas=["schedule", 0], latent_image=["first_joint", 0]),
        "first_split": node("LTXVSeparateAVLatent", av_latent=["first_sample", 0]),
        "upscaler_weight": node("LatentUpscaleModelLoader", model_name=upscaler),
        "upscale": node("LTXVLatentUpsampler", samples=["first_split", 0], upscale_model=["upscaler_weight", 0],
                        vae=["video_vae", 0]),
        "second_joint": node("LTXVConcatAVLatent", video_latent=["upscale", 0], audio_latent=["first_split", 1]),
        "second_seed": node("RandomNoise", noise_seed=13),
        "second_sample": node("SamplerCustomAdvanced", noise=["second_seed", 0], guider=["guidance", 0],
                              sampler=["sampler_choice", 0], sigmas=["schedule", 0], latent_image=["second_joint", 0]),
        "second_split": node("LTXVSeparateAVLatent", av_latent=["second_sample", 0]),
        "decode": (node("LTXVTiledVAEDecode", latents=["second_split", 0], vae=["video_vae", 0],
                        horizontal_tiles=2, vertical_tiles=2, overlap=6) if legacy_decoder
                   else node("VAEDecodeTiled", samples=["second_split", 0], vae=["video_vae", 0], tile_size=256)),
        "decode_audio": node("VAEDecodeAudio", samples=["second_split", 1], vae=["audio_vae", 0]),
    })
    return graph


def write_workflow(path, graph):
    path.write_text(json.dumps(graph), encoding="utf-8")
    return path
