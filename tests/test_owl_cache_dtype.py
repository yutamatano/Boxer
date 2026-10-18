"""Text cache compatibility across CPU/GPU and inference precision settings."""

import hashlib
from pathlib import Path

import pytest
import torch

import owl.owl_wrapper as ow


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_detector_accepts_cached_text_embeddings(
    tmp_path, monkeypatch, legacy, cache_dtype, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    checkpoint = Path(ow._CKPT_PATH)
    if not checkpoint.is_file():
        pytest.skip("OWLv2 checkpoint not installed (see README)")
    local_checkpoint = tmp_path / "model.pt"
    local_checkpoint.symlink_to(checkpoint.resolve())
    monkeypatch.setattr(ow, "_CKPT_PATH", str(local_checkpoint))

    prompts = ["chair", "table"]
    prompt_hash = hashlib.md5("\n".join(prompts).encode()).hexdigest()[:12]
    cache_path = tmp_path / f"model_textemb_{prompt_hash}.pt"
    embeddings = torch.ones(2, 512, dtype=cache_dtype)
    torch.save(
        embeddings if legacy else {"prompts": prompts, "embeddings": embeddings},
        cache_path,
    )

    wrapper = ow.OwlWrapper(
        device, text_prompts=prompts, warmup=False, precision="bfloat16"
    )
    head = wrapper.vision_detector.class_head
    dtype = next(head.parameters()).dtype
    image_embeds = torch.ones(1, 2, head.dense0.in_features, dtype=dtype, device=device)
    with torch.no_grad():
        logits, _ = head(image_embeds, wrapper.text_embeddings, wrapper.query_mask)
        expected, _ = head(
            image_embeds, embeddings.to(device=device, dtype=dtype), wrapper.query_mask
        )

    assert wrapper.text_embeddings.dtype == next(head.parameters()).dtype
    expected_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    assert wrapper.text_embeddings.dtype == expected_dtype
    torch.testing.assert_close(logits, expected)
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert (cached if legacy else cached["embeddings"]).dtype == cache_dtype


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_gpu_cache_save_and_prompt_update(tmp_path, monkeypatch, precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    checkpoint = Path(ow._CKPT_PATH)
    if not checkpoint.is_file():
        pytest.skip("OWLv2 checkpoint not installed")
    local = tmp_path / "model.pt"
    local.symlink_to(checkpoint.resolve())
    monkeypatch.setattr(ow, "_CKPT_PATH", str(local))
    prompts = ["precision cache probe"]
    wrapper = ow.OwlWrapper("cuda", text_prompts=prompts, precision=precision, warmup=False)
    cache = next(tmp_path.glob("model_textemb_*.pt"))
    saved = torch.load(cache, map_location="cpu", weights_only=False)["embeddings"]
    assert saved.dtype == torch.float32
    # Stored features must precede rounding to the GPU inference dtype.
    torch.testing.assert_close(saved, wrapper._encode_text(prompts), rtol=0, atol=0)
    wrapper.set_text_prompts(["chair", "table"])
    assert wrapper.text_embeddings.dtype == next(wrapper.vision_detector.parameters()).dtype
    assert wrapper.text_embeddings.device.type == "cuda"
    wrapper._warmup()
