import os
import sys
import types

import torch.nn as nn

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import inference
from inference import _ensure_ultralytics_backports


def test_backport_adds_missing_layers():
    class DummyC2f(nn.Module):
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
            super().__init__()
            self.c = int(c2 * e)
            self.n = n

    class DummyC3(nn.Module):
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
            super().__init__()
            self.c1 = c1
            self.c2 = c2
            self.n = n

    class DummyBottleneck(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    block_module = types.SimpleNamespace(
        C2f=DummyC2f,
        C3=DummyC3,
        Bottleneck=DummyBottleneck,
    )

    _ensure_ultralytics_backports(block_module)

    assert hasattr(block_module, "C3k")
    assert hasattr(block_module, "C3k2")

    layer = block_module.C3k2(16, 32, n=2, c3k=True)
    assert isinstance(layer.m, nn.ModuleList)
    assert len(layer.m) == 2


def test_predict_ultralytics_path(monkeypatch):
    class DummyTensor:
        def __init__(self, data):
            self._data = data

        def cpu(self):
            return self

        def tolist(self):
            return self._data

    class DummyBoxes:
        def __init__(self):
            self.xywhn = DummyTensor([[0.5, 0.5, 0.2, 0.2]])
            self.conf = DummyTensor([0.9])
            self.cls = DummyTensor([1.0])

    class DummyResult:
        def __init__(self):
            self.boxes = DummyBoxes()

    class DummyModel:
        def __init__(self):
            self.predict_calls = []

        def predict(self, image, verbose=False, device=None):
            self.predict_calls.append((image.size, verbose, device))
            return [DummyResult()]

    dummy_model = DummyModel()
    dummy_image = Image.new("RGB", (100, 100))

    monkeypatch.setattr(inference, "YOLO", object())

    lines = inference.predict(dummy_model, dummy_image, use_sahi=False)

    assert lines == [("1 0.500000 0.500000 0.200000 0.200000", 0.9)]
    assert dummy_model.predict_calls
