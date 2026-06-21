"""
NLP Integration: Speech-to-Text (Whisper) + Text Emotion Analysis.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import whisper as openai_whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False

try:
    from transformers import pipeline as hf_pipeline
    HAS_HF = True
except ImportError:
    HAS_HF = False

import numpy as np


# ─── Speech-to-Text ───────────────────────────────────────────────────────────

_whisper_model = None
_fw_model = None
_sentiment_pipeline = None


def load_whisper(model_size: str = "base") -> bool:
    global _whisper_model
    if HAS_FASTER_WHISPER:
        try:
            _whisper_model = WhisperModel(model_size, compute_type="int8")
            return True
        except Exception as e:
            logger.warning(f"FasterWhisper load failed: {e}")

    if HAS_WHISPER:
        try:
            _whisper_model = openai_whisper.load_model(model_size)
            return True
        except Exception as e:
            logger.warning(f"Whisper load failed: {e}")

    return False


def transcribe_audio(audio_path: str, model_size: str = "base") -> Dict:
    """Transcribe audio file to text."""
    global _whisper_model

    if _whisper_model is None:
        load_whisper(model_size)

    if _whisper_model is None:
        return {"text": "", "language": "en", "error": "Whisper not available"}

    try:
        if HAS_FASTER_WHISPER and hasattr(_whisper_model, "transcribe"):
            segments, info = _whisper_model.transcribe(audio_path, beam_size=5)
            text = " ".join(seg.text for seg in segments)
            return {"text": text.strip(), "language": info.language, "error": None}
        elif HAS_WHISPER:
            result = _whisper_model.transcribe(audio_path)
            return {"text": result["text"].strip(), "language": result.get("language", "en"), "error": None}
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"text": "", "language": "en", "error": str(e)}

    return {"text": "", "language": "en", "error": "No transcription model available"}


def transcribe_audio_bytes(audio_bytes: bytes, sr: int = 22050, model_size: str = "base") -> Dict:
    """Transcribe from raw audio bytes via temp file."""
    import tempfile, os, soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_bytes if isinstance(audio_bytes, np.ndarray) else
                 np.frombuffer(audio_bytes, dtype=np.float32), sr)
        result = transcribe_audio(tmp.name, model_size)

    try:
        os.unlink(tmp.name)
    except Exception:
        pass

    return result


# ─── Text Emotion Analysis ────────────────────────────────────────────────────

def load_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return True

    if not HAS_HF:
        return False

    try:
        # Lightweight text classification model
        _sentiment_pipeline = hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
        )
        return True
    except Exception:
        pass

    try:
        _sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        return True
    except Exception as e:
        logger.warning(f"Sentiment pipeline failed: {e}")
        return False


def analyze_text_emotion(text: str) -> Dict:
    """Analyze emotion in text. Returns emotion + confidence scores."""
    if not text.strip():
        return {"emotion": "neutral", "confidence": 0.5, "scores": {}, "error": "Empty text"}

    global _sentiment_pipeline

    if _sentiment_pipeline is None:
        load_sentiment_pipeline()

    if _sentiment_pipeline is not None:
        try:
            result = _sentiment_pipeline(text[:512])
            if isinstance(result[0], list):
                # All-scores format
                scores_raw = result[0]
                scores = {item["label"].lower(): item["score"] for item in scores_raw}
                best = max(scores_raw, key=lambda x: x["score"])
                return {
                    "emotion": _map_text_emotion(best["label"].lower()),
                    "confidence": best["score"],
                    "scores": {_map_text_emotion(k): v for k, v in scores.items()},
                    "error": None,
                }
            else:
                label = result[0]["label"].lower()
                score = result[0]["score"]
                return {
                    "emotion": _map_text_emotion(label),
                    "confidence": score,
                    "scores": {_map_text_emotion(label): score},
                    "error": None,
                }
        except Exception as e:
            logger.warning(f"Text emotion analysis failed: {e}")

    # Keyword fallback
    return _keyword_emotion_analysis(text)


def _map_text_emotion(label: str) -> str:
    mapping = {
        "joy": "happy", "happiness": "happy",
        "sadness": "sad", "anger": "angry",
        "fear": "fearful", "disgust": "disgust",
        "surprise": "surprise", "neutral": "neutral",
        "positive": "happy", "negative": "sad",
        "calm": "calm",
    }
    return mapping.get(label, label)


def _keyword_emotion_analysis(text: str) -> Dict:
    text_lower = text.lower()
    keyword_map = {
        "happy": ["happy", "joy", "great", "wonderful", "love", "excited", "amazing"],
        "sad": ["sad", "depressed", "unhappy", "cry", "miss", "lonely", "terrible"],
        "angry": ["angry", "furious", "hate", "frustrat", "annoy", "rage"],
        "fearful": ["afraid", "scared", "fear", "terrif", "panic", "worry"],
        "disgust": ["disgust", "gross", "awful", "horrible", "repuls"],
        "surprise": ["surprise", "wow", "unbeliev", "sudden", "shock"],
        "calm": ["calm", "relax", "peace", "quiet", "still"],
        "neutral": [],
    }

    scores = {k: 0.0 for k in keyword_map}
    for emotion, keywords in keyword_map.items():
        for kw in keywords:
            if kw in text_lower:
                scores[emotion] += 1.0

    total = sum(scores.values())
    if total == 0:
        scores["neutral"] = 1.0
        total = 1.0

    scores = {k: v / total for k, v in scores.items()}
    best = max(scores, key=scores.get)
    return {
        "emotion": best,
        "confidence": scores[best],
        "scores": scores,
        "error": None,
        "method": "keyword_fallback",
    }


# ─── Combined Analysis ────────────────────────────────────────────────────────

def combined_emotion_analysis(
    speech_emotion: str,
    speech_confidence: float,
    text_result: Dict,
) -> Dict:
    text_emotion = text_result.get("emotion", "neutral")
    text_confidence = text_result.get("confidence", 0.5)
    agreement = (speech_emotion == text_emotion)

    # Weighted combination: speech 60%, text 40%
    combined_confidence = 0.6 * speech_confidence + 0.4 * text_confidence

    return {
        "speech_emotion": speech_emotion,
        "speech_confidence": speech_confidence,
        "text_emotion": text_emotion,
        "text_confidence": text_confidence,
        "agreement": agreement,
        "combined_emotion": speech_emotion if speech_confidence >= text_confidence else text_emotion,
        "combined_confidence": combined_confidence,
        "analysis": (
            f"Speech and text analyses {'agree' if agreement else 'diverge'}. "
            f"Speech: {speech_emotion} ({speech_confidence:.0%}), "
            f"Text: {text_emotion} ({text_confidence:.0%})."
        ),
    }
