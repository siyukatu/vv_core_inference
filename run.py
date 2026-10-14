import argparse
import os
import time
from itertools import product
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import soundfile

from vv_core_inference.forwarder import Forwarder


def run(
    yukarin_s_model_dir: Path,
    yukarin_sa_model_dir: Path,
    yukarin_sosoa_model_dir: Path,
    hifigan_model_dir: Path,
    use_gpu: bool,
    use_wgpu: bool,
    use_coreml: bool,
    devices: Optional[str],
    coreml_model_format: str,
    coreml_compute_units: str,
    texts: List[str],
    speaker_ids: List[int],
    method: str,
):
    np.random.seed(0)
    device = "cpu"
    if use_gpu:
        device = "cuda"
    if use_wgpu:
        device = "wgpu"
    # 処理（duration, intonation, spectrogram, vocoder）ごとの device
    stage_devices: Dict[str, str] = {
        "duration": device,
        "intonation": device,
        "spectrogram": device,
        "vocoder": device,
    }
    if use_coreml:
        # CoreML で速くなるのは vocoder だけ。他は CPU のほうが速い（README 参照）
        stage_devices["vocoder"] = "coreml"
    if devices:
        # 例: --devices duration=cpu,intonation=cpu,spectrogram=cpu,vocoder=coreml
        for item in devices.split(","):
            stage, value = item.split("=")
            assert stage in stage_devices, f"unknown stage: {stage}"
            stage_devices[stage] = value
    print(f"devices: {stage_devices}")
    if method == "torch":
        from vv_core_inference.make_decode_forwarder import make_decode_forwarder
        from vv_core_inference.make_yukarin_s_forwarder import make_yukarin_s_forwarder
        from vv_core_inference.make_yukarin_sa_forwarder import (
            make_yukarin_sa_forwarder,
        )
    if method == "onnx":
        import onnxruntime

        from vv_core_inference.onnx_decode_forwarder import make_decode_forwarder
        from vv_core_inference.onnx_yukarin_s_forwarder import make_yukarin_s_forwarder
        from vv_core_inference.onnx_yukarin_sa_forwarder import (
            make_yukarin_sa_forwarder,
        )

        from vv_core_inference import providers

        providers.COREML_OPTIONS["ModelFormat"] = coreml_model_format
        providers.COREML_OPTIONS["MLComputeUnits"] = coreml_compute_units

        if use_gpu:
            assert onnxruntime.get_device() == "GPU", (
                "Install onnxruntime-gpu if you want to use GPU."
            )
        # if use_wgpu:
        #     assert onnxruntime.get_device() == "CPU-WEBGPU", (
        #         "Build onnxruntime with --use_webgpu if you want to use WebGPU."
        #     )

    # yukarin_s
    yukarin_s_forwarder = make_yukarin_s_forwarder(
        yukarin_s_model_dir=yukarin_s_model_dir,
        device=stage_devices["duration"] if method == "onnx" else device,
    )

    # yukarin_sa
    yukarin_sa_forwarder = make_yukarin_sa_forwarder(
        yukarin_sa_model_dir=yukarin_sa_model_dir,
        device=stage_devices["intonation"] if method == "onnx" else device,
    )

    # decoder
    if method == "onnx":
        decode_forwarder = make_decode_forwarder(
            yukarin_sosoa_model_dir=yukarin_sosoa_model_dir,
            hifigan_model_dir=hifigan_model_dir,
            device=stage_devices["spectrogram"],
            vocoder_device=stage_devices["vocoder"],
        )
    else:
        decode_forwarder = make_decode_forwarder(
            yukarin_sosoa_model_dir=yukarin_sosoa_model_dir,
            hifigan_model_dir=hifigan_model_dir,
            device=device,
        )

    # Forwarder。このForwarderクラスの中を書き換えずに
    # yukarin_s_forwarder、yukarin_sa_forwarder、decode_forwarderを置き換えたい。
    forwarder = Forwarder(
        yukarin_s_forwarder=yukarin_s_forwarder,
        yukarin_sa_forwarder=yukarin_sa_forwarder,
        decode_forwarder=decode_forwarder,
    )

    times = []
    for text, speaker_id in product(texts, speaker_ids):
        current = time.time()
        wave = forwarder.forward(
            text=text, speaker_id=speaker_id, f0_speaker_id=speaker_id
        )
        times.append(time.time() - current)
        print(
            f"method={method}, text={text}, speaker_id={speaker_id}, "
            f"elapsed={times[-1]:.3f} seconds, "
        )
        if method == "torch" or method == "onnx":
            soundfile.write(
                f"{method}-{text}-{speaker_id}.wav", data=wave, samplerate=24000
            )

    print(f"Average time for all runs: {np.mean(times):.3f} seconds")
    print(f"Average time excluding first run: {np.mean(times[1:]):.3f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yukarin_s_model_dir", type=Path, default=Path("model/yukarin_s")
    )
    parser.add_argument(
        "--yukarin_sa_model_dir", type=Path, default=Path("model/yukarin_sa")
    )
    parser.add_argument(
        "--yukarin_sosoa_model_dir", type=Path, default=Path("model/yukarin_sosoa")
    )
    parser.add_argument("--hifigan_model_dir", type=Path, default=Path("model/hifigan"))
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_wgpu", action="store_true")
    parser.add_argument(
        "--use_coreml",
        action="store_true",
        help="vocoder だけを CoreML で実行する（他の処理は CPU のほうが速いため）",
    )
    parser.add_argument(
        "--devices",
        help="処理ごとの device。例: duration=cpu,intonation=cpu,spectrogram=cpu,vocoder=coreml"
        "（--method=onnx のとき。値は cpu / cuda / wgpu / coreml）",
    )
    parser.add_argument(
        "--coreml_model_format",
        choices=["MLProgram", "NeuralNetwork"],
        default="MLProgram",
        help="CoreML のモデル形式。onnxruntime の既定値 NeuralNetwork は大幅に遅い",
    )
    parser.add_argument(
        "--coreml_compute_units",
        choices=["CPUAndGPU", "ALL", "CPUOnly", "CPUAndNeuralEngine"],
        default="CPUAndGPU",
    )
    parser.add_argument("--texts", nargs="+", default=["こんにちは、どうでしょう"])
    parser.add_argument("--speaker_ids", nargs="+", type=int, default=[5, 9])
    parser.add_argument("--method", choices=["torch", "onnx"], default="torch")
    run(**vars(parser.parse_args()))
