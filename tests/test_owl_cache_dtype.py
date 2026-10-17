"""Regression coverage for text caches reused by a CPU detector."""

import hashlib
from pathlib import Path

import pytest
import torch

import owl.owl_wrapper as ow


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_cpu_detector_accepts_cached_text_embeddings(
    tmp_path, monkeypatch, legacy, cache_dtype
):
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

    wrapper = ow.OwlWrapper("cpu", text_prompts=prompts, warmup=False)
    head = wrapper.vision_detector.class_head
    image_embeds = torch.ones(1, 2, head.dense0.in_features)
    with torch.no_grad():
        logits, _ = head(image_embeds, wrapper.text_embeddings, wrapper.query_mask)
        expected, _ = head(image_embeds, embeddings.float(), wrapper.query_mask)

    assert wrapper.text_embeddings.dtype == next(head.parameters()).dtype
    assert wrapper.text_embeddings.dtype == torch.float32
    torch.testing.assert_close(logits, expected)
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert (cached if legacy else cached["embeddings"]).dtype == cache_dtype
