from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ndarray
import onnxruntime

from vv_core_inference.providers import make_providers


def make_decode_forwarder(
    yukarin_sosoa_model_dir: Path,
    hifigan_model_dir: Path,
    device,
    convert=False,
    vocoder_device=None,
):
    # spectrogram（yukarin_sosoa）と vocoder（hifigan）で別の device を使えるようにする。
    # CoreML では spectrogram は多くの演算が未対応で分割されて遅くなるため、vocoder だけ CoreML にするとよい
    if vocoder_device is None:
        vocoder_device = device
    spectrogram_session = onnxruntime.InferenceSession(
        str(yukarin_sosoa_model_dir.joinpath("spectrogram.onnx")),
        providers=make_providers(device),
    )
    vocoder_session = onnxruntime.InferenceSession(
        str(hifigan_model_dir.joinpath("vocoder.onnx")),
        providers=make_providers(vocoder_device),
    )

    def _dispatcher(
        length: int,
        phoneme_size: int,
        f0: ndarray,
        phoneme: ndarray,
        speaker_id: Optional[ndarray] = None,
    ):
        f0 = np.asarray(f0)
        phoneme = np.asarray(phoneme)
        if speaker_id is not None:
            speaker_id = np.asarray(speaker_id)
            speaker_id = speaker_id.reshape((1,)).astype(np.int64)
        spec = spectrogram_session.run(
            ["spec"],
            {
                "f0": f0,
                "phoneme": phoneme,
                "speaker_id": speaker_id,
            },
        )[0]
        wave = vocoder_session.run(
            ["wave"],
            {
                "spec": spec,
            },
        )[0]
        return None, wave

    return _dispatcher
