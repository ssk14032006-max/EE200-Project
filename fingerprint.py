"""
fingerprint.py
Core audio fingerprinting logic: peak-picking, hashing, database building,
and matching via offset histogram -- plus the per-stage timing and ranking
info needed to drive the demo-style UI (pipeline breakdown, cluster score,
candidate ranking, alignment-spike plot).
"""

import os
import time
import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter

from audio_io import load_audio, AUDIO_EXTENSIONS

# ---- Tunable parameters (keep consistent between indexing and querying!) ----
NEIGHBORHOOD_SIZE = 10
AMP_THRESHOLD = 5
NPERSEG = 1024
TIME_WINDOW = 3.0
FAN_OUT = 15
DT_ROUND = 2
MATCH_THRESHOLD = 20  # below this score, report "no match"


def get_peaks(data, sr, nperseg=NPERSEG, neighborhood_size=NEIGHBORHOOD_SIZE,
              threshold=AMP_THRESHOLD):
    """Compute spectrogram and pick local-maxima peaks (the constellation)."""
    frequencies, times, Sxx = signal.spectrogram(data, fs=sr, nperseg=nperseg)
    data_max = maximum_filter(Sxx, size=neighborhood_size, mode='constant')
    peaks = (Sxx == data_max) & (Sxx > threshold)
    y_peaks, x_peaks = np.where(peaks)
    t_peaks, f_peaks = times[x_peaks], frequencies[y_peaks]
    return frequencies, times, Sxx, t_peaks, f_peaks


def generate_hashes(t_peaks, f_peaks, time_window=TIME_WINDOW,
                     fan_out=FAN_OUT, dt_round=DT_ROUND):
    """
    Target-zone hashing: pair each anchor peak with nearby future peaks.
    Returns list of (hash, anchor_time) where hash = (f1, f2, dt).
    dt is rounded to absorb small spectrogram time-grid alignment jitter
    between a query clip and the full song it was cut from.
    """
    hashes = []
    for i in range(len(t_peaks)):
        count = 0
        for j in range(i + 1, len(t_peaks)):
            dt = t_peaks[j] - t_peaks[i]
            if dt > time_window:
                break
            f1, f2 = f_peaks[i], f_peaks[j]
            hashes.append(((f1, f2, round(dt, dt_round)), t_peaks[i]))
            count += 1
            if count >= fan_out:
                break
    return hashes


def build_database(folder_path):
    """
    Index every supported audio file in folder_path.
    Returns:
      {
        'hashes': {hash: [(song_name, anchor_time), ...], ...},
        'songs':  {song_name: {'n_hashes', 't_peaks', 'f_peaks', 'duration'}, ...}
      }
    """
    database = {'hashes': {}, 'songs': {}}
    files = sorted(f for f in os.listdir(folder_path)
                    if f.lower().endswith(AUDIO_EXTENSIONS))

    for song_name in files:
        full_path = os.path.join(folder_path, song_name)
        data, sr = load_audio(full_path)
        _, _, _, t_peaks, f_peaks = get_peaks(data, sr)
        song_hashes = generate_hashes(t_peaks, f_peaks)

        for h, t_anchor in song_hashes:
            database['hashes'].setdefault(h, []).append((song_name, t_anchor))

        database['songs'][song_name] = {
            'n_hashes': len(song_hashes),
            't_peaks': t_peaks,
            'f_peaks': f_peaks,
            'duration': len(data) / sr,
        }

    return database


def identify_song(data, sr, database, threshold=MATCH_THRESHOLD):
    """
    data: 1D numpy array (mono audio samples), sr: sample rate.
    Returns a dict with the match result, candidate ranking, timing
    breakdown, and everything needed to plot the spectrogram,
    constellation, full-song fingerprint, and offset histogram.
    """
    timings = {}

    t0 = time.perf_counter()
    frequencies, times_arr, Sxx = signal.spectrogram(data, fs=sr, nperseg=NPERSEG)
    t1 = time.perf_counter()
    timings['spectrogram_ms'] = (t1 - t0) * 1000
    timings['spectrogram_shape'] = f"{Sxx.shape[0]}x{Sxx.shape[1]}"

    data_max = maximum_filter(Sxx, size=NEIGHBORHOOD_SIZE, mode='constant')
    peaks_mask = (Sxx == data_max) & (Sxx > AMP_THRESHOLD)
    y_peaks, x_peaks = np.where(peaks_mask)
    t_peaks, f_peaks = times_arr[x_peaks], frequencies[y_peaks]
    t2 = time.perf_counter()
    timings['constellation_ms'] = (t2 - t1) * 1000
    timings['n_peaks'] = len(t_peaks)

    query_hashes = generate_hashes(t_peaks, f_peaks)
    t3 = time.perf_counter()
    timings['hashing_ms'] = (t3 - t2) * 1000
    timings['n_hashes'] = len(query_hashes)

    offsets = defaultdict(Counter)
    db_hashes = database['hashes']
    for h, t_query in query_hashes:
        if h not in db_hashes:
            continue
        for song_name, t_db in db_hashes[h]:
            offset = round(t_db - t_query, 2)
            offsets[song_name][offset] += 1
    t4 = time.perf_counter()
    timings['db_lookup_ms'] = (t4 - t3) * 1000
    timings['n_tracks_in_db'] = len(database['songs'])

    candidates = []
    for song_name, hist in offsets.items():
        offset, count = hist.most_common(1)[0]
        candidates.append((song_name, count, offset))
    candidates.sort(key=lambda x: x[1], reverse=True)
    t5 = time.perf_counter()
    timings['scoring_ms'] = (t5 - t4) * 1000
    timings['total_ms'] = (t5 - t0) * 1000

    if candidates:
        best_song, best_score, best_offset = candidates[0]
        best_hist = offsets[best_song]
        runner_up_score = candidates[1][1] if len(candidates) > 1 else 0
    else:
        best_song, best_score, best_offset = None, 0, None
        best_hist, runner_up_score = None, 0

    matched = best_song if best_score >= threshold else None
    ratio = (best_score / runner_up_score) if runner_up_score > 0 else None

    return {
        'song': matched,
        'raw_best_song': best_song,
        'score': best_score,
        'runner_up_score': runner_up_score,
        'ratio': ratio,
        'best_offset': best_offset,
        'candidates': candidates[:5],
        'offsets': offsets,
        'best_hist': best_hist,
        'frequencies': frequencies,
        'times': times_arr,
        'Sxx': Sxx,
        't_peaks': t_peaks,
        'f_peaks': f_peaks,
        'timings': timings,
        'query_duration': len(data) / sr,
    }
