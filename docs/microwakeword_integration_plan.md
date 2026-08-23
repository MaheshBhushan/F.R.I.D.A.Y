# microWakeWord integration plan

Status: Phase 1 investigation only. This document does not change FRIDAY's
current wake behavior.

## Reference baseline

This plan was verified against the canonical Open Home Foundation repository at
commit [`4665173`](https://github.com/OHF-Voice/micro-wake-word/tree/4665173cd35f1cff9a61e06fc427f124766c488e)
and the ESPHome model collection at commit
[`05b6592`](https://github.com/esphome/micro-wake-word-models/tree/05b65922cc433c9df13e98e32a7fe520758c837e).
The former `kahrendt/microWakeWord` repository currently resolves to the same
commit as the OHF repository; OHF remains the primary reference. Piper sample
generation was checked at
[`2971426`](https://github.com/rhasspy/piper-sample-generator/tree/2971426a55072f7d22fec416ca7800df8bd23207).

Upstream explicitly describes custom model training as an early, advanced
workflow that normally needs repeated data and hyperparameter experiments. A
model emitted by the basic notebook is a baseline, not a production result.

Primary references:

- [microWakeWord README](https://github.com/OHF-Voice/micro-wake-word/blob/4665173cd35f1cff9a61e06fc427f124766c488e/README.md)
- [basic training notebook](https://github.com/OHF-Voice/micro-wake-word/blob/4665173cd35f1cff9a61e06fc427f124766c488e/notebooks/basic_training_notebook.ipynb)
- [data sources](https://github.com/OHF-Voice/micro-wake-word/blob/4665173cd35f1cff9a61e06fc427f124766c488e/documentation/data_sources.md)
- [feature frontend](https://github.com/OHF-Voice/micro-wake-word/blob/4665173cd35f1cff9a61e06fc427f124766c488e/microwakeword/audio/audio_utils.py)
- [desktop inference](https://github.com/OHF-Voice/micro-wake-word/blob/4665173cd35f1cff9a61e06fc427f124766c488e/microwakeword/inference.py)
- [training and conversion entry point](https://github.com/OHF-Voice/micro-wake-word/blob/4665173cd35f1cff9a61e06fc427f124766c488e/microwakeword/model_train_eval.py)
- [streaming evaluation](https://github.com/OHF-Voice/micro-wake-word/blob/4665173cd35f1cff9a61e06fc427f124766c488e/microwakeword/test.py)
- [Piper sample generator](https://github.com/rhasspy/piper-sample-generator/tree/2971426a55072f7d22fec416ca7800df8bd23207)

## How upstream works

### Versions and dependencies

The package metadata permits Python 3.10 or newer, while the training notebook
specifically recommends Python 3.10. The reproducible training environment will
therefore pin Python 3.10. Upstream currently declares these principal training
dependencies:

- TensorFlow 2.18 or newer
- NumPy 2.x
- `pymicro-features`
- `ai-edge-litert`
- `audiomentations`
- `audio-metadata`
- Hugging Face `datasets`
- `mmap-ninja`
- `webrtcvad-wheels`

Piper sample generator 3.2.0 supports Python 3.9 or newer and depends on
PyTorch 2.x, torchaudio, Piper TTS, NumPy 2.x, audiomentations 0.33, and
WebRTC VAD. The LibriTTS-R generator checkpoint supports speaker embedding
mixing; ordinary Piper ONNX voices can also be supplied, and multiple voices
may be cycled. Useful generation controls include speaker count/blending and
length scales.

NANI has an Intel Iris Xe GPU, not a CUDA GPU. The first reproducible path will
be a Python 3.10 CPU training environment. Training time must be measured before
deciding whether an external GPU runner is worthwhile.

### Audio and sample format

`Clips` accepts audio files selected by a directory and glob. Hugging Face
`datasets.Audio` decodes and resamples them to 16 kHz when read. We will
normalize every project-owned corpus earlier at ingestion to WAV, mono, 16 kHz,
signed 16-bit PCM. This makes validation deterministic and matches FRIDAY's
capture frames exactly.

Positive samples are individual clips containing the target phrase. Long
ambient negatives remain long recordings because false accepts/hour must be
evaluated against actual elapsed audio, not a count of derived windows.
Generated features are stored in Ragged Mmap directories under these recognized
splits:

```text
training/
validation/
testing/
validation_ambient/
testing_ambient/
```

Each split may contain one or more directories whose names end in `_mmap`.
Training YAML entries point to a feature root and declare `sampling_weight`,
`penalty_weight`, positive/negative `truth`, `truncation_strategy`, and `type`.
The native truncation strategies are `random`, `truncate_start`,
`truncate_end`, `fixed_right_cutoff`, `none`, and the ambient-only `split`
workflow.

### Feature extraction

The required native frontend is TensorFlow Lite Micro's `micro_speech`
microfrontend, exposed on desktop through `pymicro-features` or TensorFlow's
microfrontend op. It consumes 16 kHz mono PCM and produces 40 features from a
30 ms window. A feature slice is produced every 10 ms, retaining 20 ms of audio
history between slices. The frontend includes noise reduction, PCAN automatic
gain control, and the exact scaling expected by the model.

Training must call upstream `generate_features_for_clip` and
`SpectrogramGeneration`; it must not reuse openWakeWord mels or embeddings.
The C frontend is preferred because it is also the viable production path.

For every generated split the pipeline will assert and print:

- source clip count and exact WAV parameters;
- audio samples and duration computed as `sample_count / sample_rate`;
- feature tensor count, per-tensor shape, and dtype;
- the invariant `shape[1] == 40`;
- positive, ordinary-negative, and ambient duration separately;
- deterministic train/validation/test membership and counts.

Feature rows are overlapping observations, not independent audio duration.
Duration reports will be derived from source audio headers/sample counts. For
an ambient feature track, upstream's streaming metric uses
`prediction_count * model_stride * feature_step_seconds`; this is valid only
for the predictions actually emitted by streaming inference. Unit tests will
cover both calculations and a fixture that would fail under the previous 16x
row-counting error.

### Augmentation and data

The native augmentation path supports parametric EQ, distortion, pitch shift,
band-stop filtering, colored noise, background mixing at configured SNR, gain,
gain transition, and room impulse responses. It pads/truncates a target clip to
a configured duration and jitters the word near the end of the window.

Upstream names FSD50K, FMA, WHAM!, BIRD impulse responses, VOiCES, Common Voice,
and DiPCo. Their licenses differ; downloaded data and derived model eligibility
must be recorded per source. The notebook warns that its convenient mixed
download should be treated as personal/non-commercial. No YouTube audio will
be fetched without a documented lawful source and license.

Piper positives will cover `Friday` alone and the requested command contexts.
Context clips remain wake-word positives—the label means that `Friday` occurs,
not that the command is recognized. Generation will vary voices, speaker
embeddings, length scales, pronunciations, and Piper noise controls, followed
by native microWakeWord augmentation for acoustic conditions.

Real positives are mandatory. The recorder will use FRIDAY's existing
`AudioResourceManager`/device routing and emit normalized WAV plus a sidecar
manifest. A fixed seed will split by original recording identity before any
augmentation, preventing augmented siblings from crossing train, validation,
or test boundaries. Collection cannot be declared complete until the primary
user records at least 100 samples; 150–200 is preferred.

Hard negatives will include empirically discovered openWakeWord and
microWakeWord near-misses, phonetic confusers, non-invoking conversation,
licensed speech/music, room and laptop sounds, keyboard/fan noise, and captured
FRIDAY TTS. Exact TTS phrases containing `Friday` belong in a separately tagged
self-wake challenge set so results are visible rather than diluted into hours
of ambient data.

### Training, selection, and export

The notebook configuration defines the 10 ms feature step, feature roots,
weights, training stages, class weights, learning rates, batch size,
SpecAugment masks, evaluation interval, accepted clip duration, and checkpoint
selection metrics. Its baseline MixedNet example uses stride 3, so streaming
inference consumes three new feature slices and emits a probability every
approximately 30 ms.

Training is non-streaming over complete spectrograms. Upstream converts the
trained model to a stateful streaming SavedModel and then to TFLite. The final
candidate is the quantized, internal-state streaming artifact:

```text
stream_state_internal_quant/stream_state_internal_quant.tflite
```

Quantization uses representative training spectrograms. Production artifacts
will pair the `.tflite` file with immutable metadata. ESPHome v2 manifests show
the fields that matter at runtime: `probability_cutoff`, `sliding_window_size`,
`feature_step_size`, model filename/version, and tensor arena size. FRIDAY will
use the trained candidate's measured cutoff and temporal window, never copy a
threshold from an unrelated model.

Checkpoint selection will first satisfy a configured ambient false-accepts/hour
target, then maximize recall among acceptable candidates. Accuracy alone is not
a release criterion. A failed evaluation produces a failed experiment, not a
promoted model.

### Inference support and the required adapter

Upstream does provide desktop inference through `microwakeword.inference.Model`
and `ai-edge-litert`. It allocates one interpreter, zeros state tensors once,
detects quantized input/output, chunks spectrograms by the model stride, invokes
the TFLite model, and dequantizes probabilities.

It does not provide FRIDAY's needed continuous raw-PCM service. `predict_clip`
constructs features for a complete clip; calling it for every 20/80 ms capture
chunk would repeatedly reset the frontend and lose its 30 ms/10 ms overlap and
PCAN/noise state. `MicroWakeWordBackend` therefore needs a small adapter that:

1. creates one persistent `pymicro_features.MicroFrontend`;
2. accepts FRIDAY's 20 ms `AudioFrame` PCM without opening a device;
3. advances the frontend in its documented 10 ms units and retains overlap;
4. batches three feature slices when the candidate model has stride 3;
5. creates and allocates one `ai_edge_litert.Interpreter` at startup;
6. preserves its internal state tensors between invokes;
7. applies the model's quantization parameters and returns a probability;
8. resets frontend/model activation state on capture restart or preemption.

TensorFlow, datasets, PyTorch, and augmentation packages are training-only.
The production environment should contain only the minimal verified runtime
set: `ai-edge-litert`, `pymicro-features`, and the NumPy version already
compatible with FRIDAY. If `pymicro-features` cannot install cleanly in the main
environment, the backend remains unavailable with a clear startup error; it
must not silently fall back or pull TensorFlow into production.

## FRIDAY integration

### Existing contracts to preserve

`AudioCaptureService` already owns the physical microphone, normalizes input
into sequence-numbered 20 ms/16 kHz/mono/int16 `AudioFrame` values, retains a
two-second ring, and provides separate wake and STT subscriptions. It remains
unchanged. Deepgram is constructed only in `_transcribe` after a wake event, so
idle audio remains local.

The current `WakeWordDetector` consumes 80 ms chunks and contains both
openWakeWord inference and legacy handoff behavior. The live `VoiceLoop` uses
`detect_chunk`, then immediately creates an `AudioCaptureService` STT
subscription whose frozen ring and pending/live frames enter the existing
Deepgram path without a gap.

The provider-independent boundary will be introduced above engines, not below
capture:

```text
AudioCaptureService (unchanged; only microphone owner)
    -> bounded wake subscription/worker
        -> WakeDetectorBackend.process_audio(AudioFrame)
            -> OpenWakeWordBackend
            -> MicroWakeWordBackend
        -> WakeResult
        -> existing WakeDetection / VoiceLoop activation
        -> existing ring snapshot -> pending -> live Deepgram stream
```

`WakeResult` will carry `detected`, `score`, monotonic timestamp, backend,
model/version, inference latency, and triggering sequence. The existing
`WakeDetection` remains the downstream compatibility object during migration.
Neither backend may own a microphone or a second pre-roll buffer.

The initial implementation will support exactly one selected backend per live
process. The benchmark tool will fan identical offline frames to both backends.
Running both models continuously in production is unnecessary for A/B testing
and would distort idle CPU measurements.

### Worker and timing

FRIDAY capture produces 20 ms frames while a typical microWakeWord model emits
one result per 30 ms. The backend will consume sequence-numbered frames and
internally maintain only the minimal feature window. Inference will run in a
dedicated worker if measured p99 runtime can obstruct the event loop/capture
fan-out. Its queue will be bounded; overflow drops the oldest not-yet-processed
wake frame, resets streaming state because continuity was broken, and logs one
structured overflow event. It will never build an unbounded delayed audio
history.

Temporal activation uses the model metadata's probability cutoff and sliding
window size (upstream example manifests commonly use five probability windows,
but that is not a default for Friday). Cooldown is applied after an accepted
wake and measured in monotonic time. One probability spike is never sufficient.

### Configuration and observability

Backend selection will be explicit and default to the existing implementation:

```yaml
wake:
  backend: openwakeword
  openwakeword:
    models: [friday, hey_friday]
    threshold: 0.5
  microwakeword:
    model: models/friday/friday-v001.tflite
    metadata: models/friday/friday-v001.json
```

The existing environment-variable deployment can map to this configuration
while it is the project's only configuration mechanism. Source edits will not
be required to switch. Invalid/missing models fail loudly. The default is not
changed until benchmark results are reviewed and approved.

Normal logs emit initialization, candidates near activation, accepted wake,
rejection reason, cooldown, overflow, and worker errors. Every inference score
is debug-only. Detection logs include backend, model version, probability,
inference latency, capture sequence, and end-to-end detection timestamp.

## Isolated project layout

Phase 2 will create this version-controlled skeleton:

```text
tools/microwakeword/
  README.md
  requirements.txt
  configs/
  datasets/              # manifests only; raw/derived audio ignored
  scripts/
    record_friday_samples.py
    generate_samples.py
    generate_features.py
    train.py
    benchmark_wake_models.py
  artifacts/             # metadata/logs tracked; large intermediates ignored
  benchmarks/            # immutable manifests and raw result JSON
models/friday/
docs/microwakeword_results.md
```

The training venv/container is separate from `.venv`. Requirements will pin
Python-compatible versions and the three upstream commits above. Dataset
manifests record source, license, hash, duration, split, phrase/context,
speaker/session, and provenance. Raw voice recordings, third-party corpora,
credentials, and caches do not enter git.

## Reproducible execution stages and gates

Each stage receives its own logical commit and stops if its gate fails:

1. **Plan:** this document; no runtime behavior change.
2. **Training environment:** isolated Python 3.10 environment installs from a
   clean checkout and can import TensorFlow, microWakeWord, and Piper.
3. **Corpus tooling:** recording, validation, hashing, deterministic grouped
   splitting, augmentation, and duration tests pass. Human recording gate:
   at least 100 real primary-user positives, with held-out sessions.
4. **Feature generation:** native microfrontend generates `[time, 40]` tensors;
   shapes, dtypes, counts, and audio-derived durations validate.
5. **Baseline experiments:** at least three configurations produce complete
   machine-readable metadata, checkpoints, TFLite models, and streaming test
   results. Failed runs remain recorded and cannot be promoted.
6. **Runtime backend:** OpenWakeWord remains intact; microWakeWord loads behind
   the common interface and passes model, stream, temporal, cooldown, failure,
   overflow, reset, and handoff tests.
7. **Fixed-corpus benchmark:** both backends receive byte-identical,
   sequence-identical audio and report recall, false accepts/hour, false
   rejects, p50/p95/p99 detection latency, inference latency, CPU, and RAM.
8. **Live acceptance:** the primary user performs 50 normal, 20 quiet, 20
   medium-distance, and 20 fast command-attached attempts. Several hours of
   real ambient audio plus the TTS self-wake corpus provide false-accept data.
9. **Recommendation:** `docs/microwakeword_results.md` identifies the exact
   artifact/data/config, raw result files, known failures, and comparison. No
   default switch occurs without explicit approval.

## Benchmark definitions

- **Recall:** accepted wakes divided by intentional wake invocations. One
  invocation may yield at most one true accept.
- **False accepts/hour:** accepted wakes on negative audio divided by the sum
  of source negative audio samples / sample rate / 3600. Cooldown cannot erase
  the underlying raw decisions from stored benchmark traces.
- **Detection latency:** monotonic time from annotated end of the word `Friday`
  to accepted wake. Negative values, if a model fires before the annotation,
  remain visible. Report p50/p95/p99 and the annotation method.
- **Inference latency:** wall-clock duration of each model invocation after
  feature slices are ready; report p50/p95/p99.
- **CPU/RAM:** process CPU and resident-set delta during a fixed-duration idle
  replay and during live idle capture, with sample interval and baseline
  recorded.
- **False rejection rate:** one minus recall on the positive corpus.

The comparison tool reads the corpus once and sends the same PCM frames and
sequence numbers to both backends. It preserves per-frame scores, decisions,
latencies, annotations, configuration, model hashes, git commit, and machine
information as JSON. Summary tables are derived artifacts, never the only
record.

## Required automated coverage

The implementation is incomplete until tests cover model loading and invalid
models, PCM validation, incremental frontend state, quantization, threshold and
activation windows, cooldown and repeated phrases, backend selection, bounded
queue overflow/reset, worker exception recovery, capture ring interaction,
wake-to-STT ordering, immediate short commands, shutdown/restart, microphone
preemption, and inference exceptions. Existing tests must remain green.

The decisive handoff regression delays STT connection by 500 ms and supplies
`Friday, are you working?` continuously. Deepgram must receive the full
sequence exactly once. This remains an audio-pipeline test independent of wake
model accuracy.

## Known constraints and open gates

- No truthful real-user recall or live latency result exists until the primary
  user records and performs the specified trials.
- No truthful false-accepts/hour claim exists until several hours of retained,
  licensed/local ambient audio have been evaluated.
- Training on this laptop is CPU-bound unless a compatible remote GPU is
  explicitly authorized; experiment duration is currently unknown.
- Dataset licenses can constrain redistribution of audio and possibly derived
  models. Every source must be reviewed before publishing an artifact.
- A TFLite file is not accepted merely because it loads. It needs matching
  metadata, hashes, dataset/config provenance, streaming benchmark results, and
  the false-accept gate.
- Production remains on OpenWakeWord until the final comparison is presented
  and an explicit backend switch is approved.
