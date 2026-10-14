# vv_core_inference

VOICEVOX のコア内で用いられているディープラーニングモデルの推論コード。VOICEVOX コア用の onnx モデルを制作できる。

サンプルとして配布しているモデルは実際の VOICEVOX のものではなく、ほとんどノイズと変わらない音が生成されます。
含まれている config の値（層の数など）は仮の値で、VOICEVOX で使用されるモデルとは異なることがあります。

## 公開している意図

VOICEVOX コアでの音声合成をより高速・軽量にするための手法の議論や提案を受けられるようにするためです。

VOICEVOX コアはこのリポジトリで作った onnx モデルを用いて計算処理（推論）が行われています。
onnx モデルをそのまま改良するのはかなり専門的な知識が必要なので、より多くの方に馴染みのある pytorch のモデルとコードを公開しています。

## 環境構築

[uv](https://github.com/astral-sh/uv) で開発しています。

## pytorch モデルのダウンロード

```bash
curl -L -o model.zip https://github.com/Hiroshiba/vv_core_inference/releases/download/0.0.3/model.zip
unzip model.zip
rm model.zip
```

## 実行

```bash
# 生成される音声はほぼノイズで、かろうじて母音がわかる程度だと思います
uv run run.py \
  --yukarin_s_model_dir "model/yukarin_s" \
  --yukarin_sa_model_dir "model/yukarin_sa" \
  --yukarin_sosoa_model_dir "model/yukarin_sosoa" \
  --hifigan_model_dir "model/hifigan" \
  --speaker_ids 5 \
  --texts "おはようございます、こんにちは、こんばんは、どうでしょうか"
```

## CoreML（macOS）で実行する

`--method=onnx --use_coreml` で、ボコーダー（hifigan の `vocoder.onnx`）だけを CoreML で実行します。
音素長・抑揚・スペクトログラムは CPU のままです（これらは CoreML で速くならないため）。

```bash
uv run run.py --yukarin_s_model_dir "model" --yukarin_sa_model_dir "model" --yukarin_sosoa_model_dir "model" --hifigan_model_dir "model" \
  --speaker_ids 1 2 3 4 5 --method=onnx --use_coreml
```

- `--devices duration=cpu,intonation=cpu,spectrogram=cpu,vocoder=coreml` のように、処理ごとに device（cpu / cuda / wgpu / coreml）を指定できます。
- `--coreml_model_format`（既定 `MLProgram`）と `--coreml_compute_units`（既定 `CPUAndGPU`）で CoreML の設定を変えられます。

### CoreML が遅くなっていた理由

onnxruntime の CoreML EP をオプションなしで使うと、`ModelFormat=NeuralNetwork`（古い形式）になります。この形式ではボコーダーが CPU より大幅に遅くなりました。
`ModelFormat=MLProgram` を指定すると、ボコーダーは CPU の約 4〜5 倍、WebGPU の約 1.7 倍の速さになりました。

また spectrogram（yukarin_sosoa）は、CoreML で実行できない演算が多く（346 ノード中 123、27 分割）、CPU との行き来で遅くなります。
全体を CoreML にすると遅くなる主な原因はこの 2 点でした。

### 計測結果（Apple M1、8 コア、16GB）

上の例と同じ長い文（約 1,200 フレーム）で、話者 1〜5 の平均（初回を除く）です。onnxruntime 1.23.0（uv.lock のもの）を使いました。

| 構成 | 1 回あたり | CPU との比 |
|---|---:|---:|
| すべて CPU | 3.66 秒 | 1.0 倍 |
| すべて CoreML（オプションなし） | 11.50 秒 | 0.32 倍 |
| decode（spectrogram + vocoder）だけ CoreML（オプションなし） | 9.43 秒 | 0.39 倍 |
| vocoder だけ CoreML（オプションなし） | 6.22 秒 | 0.59 倍 |
| **vocoder だけ CoreML（MLProgram・CPUAndGPU）** | **0.88 秒** | **4.2 倍** |
| vocoder だけ CoreML（MLProgram・ALL） | 0.89 秒 | 4.1 倍 |
| vocoder だけ CoreML（MLProgram・CPUOnly） | 1.67 秒 | 2.2 倍 |
| vocoder だけ CoreML（MLProgram・CPUAndNeuralEngine） | 1.68 秒 | 2.2 倍 |

MLProgram・CPUAndGPU の出力は、CPU の出力と比べて SN 比 約 90dB（ほぼ同一）でした。

ボコーダー単体（1,200 フレーム、onnxruntime-webgpu 1.27.0）:

| 実行方法 | 1 フレームあたり |
|---|---:|
| CPU（4 スレッド） | 3.36ms |
| WebGPU | 1.10ms |
| CoreML（オプションなし = NeuralNetwork） | 41.5ms |
| CoreML（MLProgram・CPUAndGPU） | 0.64ms |
| CoreML（MLProgram・CPUOnly） | 1.42ms |

補足:

- Neural Engine を指定しても速くなりませんでした（CPUOnly と同程度）。
- 入力の長さを固定する設定（`RequireStaticInputShapes=1`）にすると、onnxruntime が「畳み込みの入力長は 16,384 まで」という制限を確認し、
  フレーム数 × 256 がこれを超える部分を CPU に回します（64 フレーム超で発生）。長さを固定しない既定の設定では、
  長い文でもボコーダー全体が CoreML で動きました。
- 初回の実行時に CoreML 用の変換が行われるため、最初の 1 回は遅くなります。

## モデルを onnx に変換

- `uv run convert.py --yukarin_s_model_dir "model/yukarin_s" --yukarin_sa_model_dir "model/yukarin_sa" --yukarin_sosoa_model_dir "model/yukarin_sosoa" --hifigan_model_dir "model/hifigan"` で onnx への変換が可能。model フォルダ内の yukarin_s, yukarin_sa, yukarin_sosoa フォルダに onnx が保存される。

  - `speaker_ids`オプションに指定する数値は自由。どの数値を指定しても生成される onnx モデルは全ての`speaker_id`に対応しており、値を変えて実行しなおしたり、複数の id を指定したりする必要は無い。
  - yukarin_sosoa フォルダには hifi_gan と合わせた`decode.onnx`が保存される

- onnx で実行したい場合は`run.py`を`--method=onnx`で実行する； `uv run run.py --yukarin_s_model_dir "model" --yukarin_sa_model_dir "model" --yukarin_sosoa_model_dir "model" --hifigan_model_dir "model"  --speaker_ids 5  --method=onnx`
  - `speaker_ids`に複数の数値を指定すれば、通常実行と同様に各話者の音声が保存される。

## ファイル構造

- `run.py` ･･･ 音声合成のためのエントリーポイント
- `convert.py` ･･･ モデル変換のためのエントリーポイント
- `vv_core_inference` ･･･ いろいろな処理
  - `forwarder.py`
    - VOICEVOX と同じインターフェースと処理。
    - この`Forwarder`クラスを一切変更することなく、3 つの`forwarder`を与えられると完璧
  - `make_yukarin_s_forwarder.py`
    - 音素ごとの長さを求めるモデル`yukarin_s`用の`forwarder`を作る
  - `make_yukarin_sa_forwarder.py`
    - モーラごとの音高を求めるモデル`yukarin_sa`用の`forwarder`を作る
  - `make_yukarin_sosoa_forwarder.py`
    - `make_decode_forwarder`に必要な`yukarin_sosoa`用の`forwarder`を作る
  - `make_decode_forwarder.py`
    - 音声波形生成用の`forwarder`を作る
  - `onnx_yukarin_s_forwarder.py`
    - onnxruntime で動作する`yukarin_s`用の`forwarder`を作る
  - `onnx_yukarin_sa_forwarder.py`
    - onnxruntime で動作する`yukarin_sa`用の`forwarder`を作る
  - `onnx_decode_forwarder.py`
    - onnxruntime で動作する音声波形生成用の`forwarder`を作る
    - `yukarin_sosoa`も内部に組み込まれている
  - `acoustic_feature_extractor.py`
    - 音素情報やリサンプリング手法などが入っている。ディープラーニングとは関係ない。
  - `full_context_label.py`
    - フルコンテキストラベルの処理が入っている。ディープラーニングとは関係ない。
  - `utility.py`
    - 便利関数が多少ある

## 自分で学習したモデルの onnx を作りたい場合

VOICEVOX をビルドするには以下の 4 つの onnx が必要です。

- predict_duration.onnx
  - 入力
    - phoneme_list
      - shape: [sequence]
      - dtype: int
      - 値は音素 id
    - speaker_id
      - shape: [1]
      - dtype: int
  - 出力
    - phoneme_length
      - shape: [sequence]
      - dtype: float
      - 値は秒単位の継続長
- predict_intonation.onnx
  - 入力
    - length
      - shape: int
      - 値は音素 id
    - vowel_phoneme_list
      - shape: [length]
      - dtype: int
      - 値は音素 id
    - consonant_phoneme_list
      - shape: [length]
      - dtype: int
      - 値は音素 id
    - start_accent_list
      - shape: [length]
      - dtype: int
      - 値は 0 か 1
    - end_accent_list
      - shape: [length]
      - dtype: int
      - 値は 0 か 1
    - start_accent_phrase_list
      - shape: [length]
      - dtype: int
      - 値は 0 か 1
    - end_accent_phrase_list
      - shape: [length]
      - dtype: int
      - 値は 0 か 1
    - speaker_id
      - shape: [1]
      - dtype: int
  - 出力
    - f0_list
      - shape: [sequence]
      - dtype: float
- predict_spectrogram.onnx
  - 入力
    - f0
      - shape: [length, 1]
      - dtype: float
    - phoneme
      - shape: [length, vocab_size]
      - dtype: int
      - onehot で表現された音素
    - speaker_id
      - shape: [1]
      - dtype: int
  - 出力
    - spec
      - shape: [length, feat_size]
      - dtype: float
      - 音声の中間表現
- vocoder.onnx
  - 入力
    - f0
      - shape: [length, 1]
      - dtype: float
    - spec
      - shape: [length, feat_size]
      - dtype: float
      - 音声の中間表現
  - 出力
    - wave
      - shape: [outlength]
      - dtype: float
      - 値は [-1.0, 1.0] の音声波形
      - サンプリング周波数は 24kHz

音素 id は辞書に依存します。また predict_duration.onnx や predict_intonation.onnx の出力はコアによって変換されて predict_spectrogram.onnx や vocoder.onnx の入力になります。コアを変更しない場合は phoneme_length を元に f0 と phoneme と spec が 93.75(=24k/256)Hz になるように変換されます。predict_spectrogram.onnx は受容野が広くて逐次計算できないモデルを、vocoder.onnx は受容野が狭くて逐次計算できるモデルを想定しています。

## フォーマット・リント

```bash
# フォーマット
ruff format
ruff check --fix

# リント
ruff check
```
