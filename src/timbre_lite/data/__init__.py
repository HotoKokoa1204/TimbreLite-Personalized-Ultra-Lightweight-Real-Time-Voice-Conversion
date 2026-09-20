"""Audio data curation, ingestion, and preprocessing modules."""

from timbre_lite.data.curation import (
    AudioSegmenter,
    CurationConfig,
    DualRateSample,
    curate_source_speaker_audio,
)
from timbre_lite.data.hu_tao import (
    HuTaoDatasetConfig,
    curate_hu_tao_dataset,
    download_hu_tao_dataset,
)
from timbre_lite.data.loader import load_audio, save_wav
from timbre_lite.data.speech_filter import (
    SpeechVerificationResult,
    SpeechVerifier,
)

__all__ = [
    "AudioSegmenter",
    "CurationConfig",
    "DualRateSample",
    "HuTaoDatasetConfig",
    "SpeechVerificationResult",
    "SpeechVerifier",
    "curate_hu_tao_dataset",
    "curate_source_speaker_audio",
    "download_hu_tao_dataset",
    "load_audio",
    "save_wav",
]
