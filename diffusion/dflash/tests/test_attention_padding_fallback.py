"""Padding dispatch without flash_attn: tensor layout, gradients and backend choice."""
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def padding(monkeypatch):
    # Import real source files without loading unrelated Ray/protocol dependencies.
    device = ModuleType('verl.utils.device')
    device.is_torch_npu_available = lambda **kwargs: False
    helper = load_file('torch_padding_under_test', ROOT / 'verl/verl/utils/npu_flash_attn_utils.py')
    monkeypatch.setitem(sys.modules, 'verl.utils.device', device)
    monkeypatch.setitem(sys.modules, 'verl.utils.npu_flash_attn_utils', helper)
    module = load_file('attention_padding_under_test', ROOT / 'verl/verl/utils/attention_utils.py')
    original = builtins.__import__
    attempts = []
    def without_flash(name, *args, **kwargs):
        if name == 'flash_attn.bert_padding':
            attempts.append(name)
            raise ModuleNotFoundError("No module named 'flash_attn'", name='flash_attn')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_flash)
    return module, helper, attempts


@pytest.mark.parametrize('mask_dtype', [torch.bool, torch.int64])
def test_padding_roundtrip_and_gradients(padding, mask_dtype):
    module, _, attempts = padding
    x = torch.randn(3, 6, 2, 3, requires_grad=True)
    mask = torch.tensor([[0,1,1,1,0,0], [1,1,0,0,0,0], [0,0,0,0,0,0]], dtype=mask_dtype)
    values, indices, offsets, maximum, used = module.unpad_input(x, mask)
    torch.testing.assert_close(values, x[mask.bool()])
    assert indices.tolist() == [1,2,3,6,7]
    assert offsets.tolist() == [0,3,5,5] and offsets.dtype == torch.int32
    assert used.tolist() == [3,2,0] and maximum == 3
    padded = module.pad_input(values, indices, batch=3, seqlen=6)
    torch.testing.assert_close(padded, x * mask.bool()[:, :, None, None])
    padded.sum().backward()
    torch.testing.assert_close(x.grad, mask.bool()[:, :, None, None].expand_as(x).float())
    # This is the jagged construction used in left_right_2_no_padding.
    nested = torch.nested.nested_tensor_from_jagged(values, offsets=offsets)
    assert nested.unbind()[0].shape == (3,2,3)
    assert nested.unbind()[2].shape == (0,2,3)
    assert len(attempts) == 1  # cached backend; no repeated failed import per operation


def test_all_padding_and_unused_slots(padding):
    module, _, _ = padding
    x = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.zeros(2, 4, dtype=torch.int64)
    values, indices, offsets, maximum, used = module.unpad_input(x, mask)
    assert values.shape == (0,3) and maximum == 0
    assert offsets.tolist() == [0,0,0]
    padded = module.pad_input(values, indices, 2, 4)
    assert not padded.any()
    padded.sum().backward()
    assert not x.grad.any()
    mask[0,1] = 1
    unused = torch.zeros_like(mask)
    unused[0,2] = 1
    values, indices, offsets, maximum, used = module.unpad_input(x, mask, unused_mask=unused)
    assert indices.tolist() == [1,2]
    assert offsets.tolist() == [0,2,2] and used.tolist() == [1,0]
    torch.testing.assert_close(values, x[0,1:3])


def test_installed_flash_helpers_remain_preferred(padding, monkeypatch):
    module, helper, _ = padding
    original = builtins.__import__
    def with_flash(name, *args, **kwargs):
        if name == 'flash_attn.bert_padding':
            return helper
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', with_flash)
    assert module._get_attention_functions() == (
        helper.index_first_axis, helper.pad_input, helper.rearrange, helper.unpad_input)


def test_transitive_dependency_failure_is_not_hidden(padding, monkeypatch):
    module, _, _ = padding
    original = builtins.__import__
    def broken_flash(name, *args, **kwargs):
        if name == 'flash_attn.bert_padding':
            raise ModuleNotFoundError('missing dependency', name='some_flash_dependency')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', broken_flash)
    with pytest.raises(ModuleNotFoundError) as error:
        module._get_attention_functions()
    assert error.value.name == 'some_flash_dependency'
