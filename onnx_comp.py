from pathlib import Path

import onnx
import timm
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "corner_training_output" / "checkpoints" / "best_corners.ckpt"
ONNX_PATH = ROOT / "mobilenet.onnx"
IMG_SIZE = 640
OPSET_VERSION = 17


class MobileNetV3Corners(nn.Module):
    def __init__(self, num_keypoints: int = 4):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_large_075",
            pretrained=False,
            num_classes=num_keypoints * 2,
            exportable=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CornersExportModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint) -> dict:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    keys = ", ".join(list(checkpoint.keys())[:10])
    raise KeyError(f"Could not find a model state_dict in checkpoint. Top-level keys: {keys}")


def strip_known_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def load_model_weights(model: nn.Module, state_dict: dict) -> None:
    target_state = model.state_dict()
    cleaned_state = {}
    skipped_shape = []

    for key, value in state_dict.items():
        normalized_key = strip_known_prefixes(key)
        candidates = [normalized_key]
        if not normalized_key.startswith("backbone."):
            candidates.append(f"backbone.{normalized_key}")

        for candidate in candidates:
            if candidate not in target_state:
                continue
            if target_state[candidate].shape != value.shape:
                skipped_shape.append(candidate)
                break
            cleaned_state[candidate] = value
            break

    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing or unexpected or skipped_shape:
        raise RuntimeError(
            "Checkpoint is not compatible with the export model. "
            f"Missing: {missing[:10]}, unexpected: {unexpected[:10]}, "
            f"shape mismatches: {skipped_shape[:10]}"
        )


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = MobileNetV3Corners(num_keypoints=4)
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    load_model_weights(model, extract_state_dict(checkpoint))
    model.eval()

    export_model = CornersExportModel(model).eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        export_model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["corners"],
        dynamic_axes={
            "input": {0: "batch"},
            "corners": {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print(f"ok: exported {ONNX_PATH}")


if __name__ == "__main__":
    main()
