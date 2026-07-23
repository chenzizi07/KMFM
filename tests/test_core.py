import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kmfm.data import HSIPatchDataset
from kmfm.metrics import classification_metrics
from kmfm.model import LASSFNet, SpectralConv1D
from kmfm.splits import TRAIN, VAL, make_random_pixel_split


def test_metrics_are_from_one_confusion_matrix():
    metrics, confusion = classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), num_classes=2
    )
    assert confusion.tolist() == [[1, 1], [0, 2]]
    assert metrics["oa"] == pytest.approx(0.75)
    assert metrics["aa"] == pytest.approx(0.75)


def test_random_split_is_disjoint_and_fixed_count():
    labels = np.tile(np.arange(1, 4), (20, 20))[:, :20]
    split = make_random_pixel_split(labels, train_per_class=5, val_per_class=3, seed=9)
    assert not np.any(split.train_mask & split.val_mask)
    assert not np.any(split.train_mask & split.test_mask)
    assert split.metadata["counts"]["train"] == [5, 5, 5]
    assert split.metadata["counts"]["val"] == [3, 3, 3]


def test_context_guard_zeros_other_regions():
    cube = np.ones((5, 5, 4), dtype=np.float32)
    labels = np.ones((5, 5), dtype=np.int64)
    centers = np.zeros((5, 5), dtype=bool)
    centers[2, 2] = True
    regions = np.full((5, 5), VAL, dtype=np.int8)
    regions[1:4, 1:4] = TRAIN
    dataset = HSIPatchDataset(
        cube,
        labels,
        centers,
        patch_size=5,
        region_map=regions,
        region_value=TRAIN,
        allow_full_context=False,
    )
    patch, _, _, visible = dataset[0]
    assert visible.sum().item() == 9
    assert patch[:, 0, 0].abs().sum().item() == 0
    assert patch[:, 2, 2].sum().item() == 4


def test_spectral_conv_uses_band_length():
    encoder = SpectralConv1D(hidden_dim=8, kernels=(3, 5), branch_dim=4)
    spectrum = torch.randn(2, 17, requires_grad=True)
    output = encoder(spectrum)
    assert output.shape == (2, 8)
    output.sum().backward()
    assert spectrum.grad is not None


def test_model_forward_backward():
    model = LASSFNet(bands=12, num_classes=3, hidden_dim=16)
    patch = torch.randn(2, 12, 7, 7)
    visible = torch.ones(2, 7, 7)
    output = model(patch, visible)
    assert output["logits"].shape == (2, 3)
    loss = output["logits"].sum()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
