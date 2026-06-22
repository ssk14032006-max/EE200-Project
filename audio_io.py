"""
audio_io.py
Loads any common audio format (wav, mp3, flac, ogg, m4a) into a mono
float32 numpy array normalized to [-1, 1], using pydub (which wraps
ffmpeg). This is what lets the app accept the same format range shown
in the demo video, instead of being limited to .wav like scipy.io.wavfile.

Requires ffmpeg to be installed on the system (see packages.txt for
Streamlit Community Cloud deployment).
"""

import numpy as np
from pydub import AudioSegment

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')


def load_audio(file_obj_or_path, filename=None):
    """
    file_obj_or_path: a filesystem path (str) OR a file-like object
        (e.g. Streamlit's UploadedFile).
    filename: required when passing a file-like object, so pydub knows
        which decoder to use (inferred from the extension).
    Returns (data, sr): mono float32 array in [-1, 1], sample rate.
    """
    fmt = None
    if filename is not None:
        fmt = filename.rsplit('.', 1)[-1].lower()
    elif isinstance(file_obj_or_path, str):
        fmt = file_obj_or_path.rsplit('.', 1)[-1].lower()

    audio = AudioSegment.from_file(file_obj_or_path, format=fmt)
    audio = audio.set_channels(1)
    sr = audio.frame_rate

    # NOTE: deliberately NOT normalized to [-1, 1]. We keep raw sample
    # magnitudes (same range as scipy.io.wavfile.read's int16 output,
    # e.g. roughly +/-32768) because AMP_THRESHOLD and other fingerprint
    # parameters were tuned against that scale. Normalizing here would
    # silently push every peak below threshold and zero out the database.
    data = np.array(audio.get_array_of_samples()).astype(np.float32)
    return data, sr
