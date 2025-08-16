import importlib
from types import SimpleNamespace
import os
import pytest
np = pytest.importorskip("numpy")
import torch
from PIL import Image

# Patch model loading before importing the module
import llava.model.builder as builder

def dummy_load_pretrained_model(*args, **kwargs):
    class DummyTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            if isinstance(ids, (list, tuple)):
                return " ".join(f"tok{_id}" for _id in ids)
            return f"tok{ids}"

    class DummyModel:
        def __init__(self):
            self.device = "cpu"
            self.config = SimpleNamespace(image_grid_pinpoints=[(64, 64)])
        def eval(self):
            pass
        def __call__(self, **kwargs):
            seq_len = kwargs.get("input_ids", torch.zeros(1,1)).shape[1]
            logits = torch.randn(1, seq_len, 5)
            attentions = (torch.ones(1, 1, seq_len + 4, seq_len + 4),)
            return SimpleNamespace(logits=logits, attentions=attentions)

    class DummyProcessor:
        pass

    return DummyTokenizer(), DummyModel(), DummyProcessor(), 10

builder.load_pretrained_model = dummy_load_pretrained_model

# Import the module under test
eai = importlib.import_module('docs.extract_attention_interactive')


def test_fix_wsl_paths_windows():
    path = r"C:\\Users\\test\\file.txt"
    assert eai.fix_wsl_paths(path) == "/mnt/c/Users/test/file.txt"


def test_fix_wsl_paths_passthrough():
    path = "/mnt/c/Users/test/file.txt"
    assert eai.fix_wsl_paths(path) == path


def test_load_mask_from_file(tmp_path):
    mask = np.ones((2, 2), dtype=np.uint8)
    file_path = tmp_path / "mask.npy"
    np.save(file_path, mask)
    loaded = eai.load_mask_from_file(str(file_path))
    assert np.array_equal(loaded, mask)


@pytest.mark.parametrize("generation,attention,expected", [
    ({"max_new_tokens": 10}, {"create_collage": False}, 10),
])
def test_prepare_configs(generation, attention, expected):
    gen, attn = eai._prepare_configs(generation, attention)
    assert gen["max_new_tokens"] == expected
    assert attn["create_collage"] is False


def test_setup_output_directories(tmp_path):
    dirs = eai._setup_output_directories(tmp_path)
    for d in dirs:
        assert os.path.isdir(d)


def test_save_raw_attention_tensor(tmp_path):
    attn = np.ones((2,2))
    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()
    eai.save_raw_attention_tensor(attn, None, tensor_dir, 1, "tok", 0)
    files = list(tensor_dir.glob("*.pt"))
    assert files, "tensor file not saved"


def test_generate_next_token_greedy():
    dummy_tokenizer = SimpleNamespace(decode=lambda ids: f"tok{ids[0]}")
    class DummyModel:
        def __call__(self, **kwargs):
            logits = torch.tensor([[[0.1,0.2,0.9]]])
            return SimpleNamespace(logits=logits)
    ids, text, _ = eai._generate_next_token({}, DummyModel(), dummy_tokenizer, {"do_sample": False})
    assert ids.item() == 2
    assert text == "tok2"


def test_extract_and_process_attention(tmp_path):
    logits = torch.zeros(1,1,3)
    attentions = (torch.ones(1,1,5,5),)
    outputs = SimpleNamespace(logits=logits, attentions=attentions)
    image = Image.new("RGB", (2,2))
    attn_config = {"visualize_attn_overlays": False, "save_tensors": False}
    processed, attn_map = eai._extract_and_process_attention(
        outputs, torch.tensor([1]), "tok", 0, 4, 2, 1, image,
        attn_config, tmp_path, tmp_path, tmp_path)
    assert processed is None
    assert attn_map.shape == (2,2)
