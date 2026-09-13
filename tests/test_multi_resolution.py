import random

import torch

from deepfense.training.multi_resolution import MultiResolutionCropper, build_multi_resolution_cropper


def test_cropper_keeps_one_duration_for_the_whole_batch():
    cropper = MultiResolutionCropper(
        {
            "enabled": True,
            "sample_rate": 16000,
            "durations_sec": [0.5, 1.0, 2.0, 3.0, 4.0],
            "random_crop": False,
        }
    )
    x = torch.randn(4, 64000)
    mask = torch.ones(4, 64000)

    random.seed(0)
    x_out, mask_out = cropper.apply(x, mask)

    assert x_out.shape[0] == 4
    assert x_out.shape[1] == mask_out.shape[1]
    assert x_out.shape[1] in {8000, 16000, 32000, 48000, 64000}
    assert cropper.last_duration_samples == x_out.shape[1]
    torch.testing.assert_close(x_out, x[:, : x_out.shape[1]])


def test_cropper_random_crop_is_a_contiguous_slice():
    cropper = MultiResolutionCropper(
        {
            "enabled": True,
            "sample_rate": 16000,
            "durations_samples": [16000],
            "random_crop": True,
        }
    )
    x = torch.arange(64000, dtype=torch.float32).unsqueeze(0).repeat(3, 1)

    random.seed(1)
    x_out, _ = cropper.apply(x, None)
    assert x_out.shape == (3, 16000)
    # All rows share the same crop window.
    assert torch.equal(x_out[0], x_out[1])
    diffs = x_out[0, 1:] - x_out[0, :-1]
    assert torch.allclose(diffs, torch.ones_like(diffs))


def test_full_length_is_a_no_op():
    cropper = MultiResolutionCropper(
        {
            "enabled": True,
            "durations_samples": [64000],
            "sample_rate": 16000,
        }
    )
    x = torch.randn(2, 64000)
    x_out, mask_out = cropper.apply(x, torch.ones(2, 64000))
    assert x_out.shape == x.shape
    assert torch.equal(x_out, x)
    assert mask_out.shape == (2, 64000)


def test_disabled_builder_returns_none():
    assert build_multi_resolution_cropper(None) is None
    assert build_multi_resolution_cropper({"enabled": False}) is None


def test_sample_rate_clamps_to_available_length():
    cropper = MultiResolutionCropper(
        {
            "enabled": True,
            "sample_rate": 16000,
            "durations_sec": [4.0],
            "random_crop": False,
        }
    )
    x = torch.randn(1, 32000)
    x_out, _ = cropper.apply(x)
    assert x_out.shape[-1] == 32000
