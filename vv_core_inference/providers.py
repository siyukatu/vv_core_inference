"""onnxruntime の実行プロバイダー（EP）の指定をまとめる。

CoreML は、onnxruntime の既定設定（ModelFormat=NeuralNetwork）だと CPU より大幅に遅くなる。
ModelFormat=MLProgram を指定すると、ボコーダー（hifigan）が CPU の数倍速くなった（README 参照）。
"""

from typing import Dict, List, Union

# CoreML EP に渡すオプション。run.py の --coreml_model_format / --coreml_compute_units で変えられる
COREML_OPTIONS: Dict[str, str] = {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "CPUAndGPU",
}


def make_providers(device: str) -> List[Union[str, tuple]]:
    """device（cpu / cuda / wgpu / coreml）から InferenceSession に渡す providers を作る。"""
    providers: List[Union[str, tuple]] = ["CPUExecutionProvider"]
    if device == "cuda":
        providers.insert(0, "CUDAExecutionProvider")
    elif device == "wgpu":
        providers.insert(0, "WebGpuExecutionProvider")
    elif device == "coreml":
        providers.insert(0, ("CoreMLExecutionProvider", dict(COREML_OPTIONS)))
    elif device != "cpu":
        raise ValueError(f"unknown device: {device}")
    return providers
