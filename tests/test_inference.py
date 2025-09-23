import types

import torch.nn as nn

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
