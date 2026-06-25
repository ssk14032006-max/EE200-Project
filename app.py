"""
app.py
Streamlit app for Q3B -- "Zapp tain America".

Three tabs, matching the demo video:
  - Library : browse the indexed songs (constellation thumbnail + hash count)
  - Identify: upload (or try a sample) one query clip; see the pipeline
              timing breakdown, the match verdict with cluster score, the
              candidate ranking, and three explanatory plots (spectrogram +
              constellation, full-song fingerprint with the query window
              highlighted, and the annotated offset-histogram spike).
  - Batch   : upload many query clips, get a results.csv (filename, prediction)

Run locally:   streamlit run app.py
Deploy: push this folder (incl. database.pkl) to GitHub, connect at
share.streamlit.io. packages.txt installs ffmpeg (needed for mp3/m4a).
"""

import os
import pickle

def reassemble_database():
    # Only reassemble if the full database doesn't already exist
    if not os.path.exists('database.pkl'):
        print("Reassembling database parts...")
        # Use a list of the parts in order
        parts = ['database.pkl.partaa', 'database.pkl.partab', 'database.pkl.partac', 'database.pkl.partad', 'database.pkl.partae', 'database.pkl.partaf']
        
        with open('database.pkl', 'wb') as outfile:
            for part in parts:
                with open(part, 'rb') as infile:
                    outfile.write(infile.read())
        print("Reassembly compl ete.")

# Run this before loading the database
reassemble_database()

# Now load the reassembled file
with open('database.pkl', 'rb') as f:
    database = pickle.load(f)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from audio_io import load_audio, AUDIO_EXTENSIONS
from fingerprint import identify_song

# ----------------------------------------------------------------------
# Page setup + dark/teal styling (matches the demo video's look)
# ----------------------------------------------------------------------
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")

ACCENT = "#2dd4bf"
ACCENT_ORANGE = "#fb923c"
BG = "#0a0e0c"
PANEL = "#111815"
MUTED = "#6b7280"
WHITE = "#ffffff  "

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": "#374151",
    "axes.labelcolor": "#e5e7eb",
    "xtick.color": "#9ca3af",
    "ytick.color": "#9ca3af",
    "text.color": "#e5e7eb",
    "grid.color": "#1f2937",
    "font.family": "monospace",
    "font.size": 9,
})

SAMPLES_DIR = "samples"

st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
  <div style="font-size:28px;font-weight:800;">EE<span style="color:{ACCENT};">200</span>: Audio Fingerprinting</div>
</div>
<div style="color:{MUTED};font-size:11px;letter-spacing:2px;margin-bottom:6px;">SIGNALS, SYSTEMS &amp; NETWORKS · PROJECT</div>
<div style="color:#9ca3af;font-size:14px;margin-bottom:18px;">Index a library of songs as spectrogram fingerprints, then identify any short clip against it.</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Load the prebuilt database (built once via build_database.py)
# ----------------------------------------------------------------------
@st.cache_resource
def load_database():
    with open("database.pkl", "rb") as f:
        return pickle.load(f)


database = load_database()

library_tab, identify_tab, batch_tab = st.tabs(["◆  LIBRARY", "⚙  IDENTIFY", "▤  BATCH"])


# ----------------------------------------------------------------------
# Shared plotting helpers
# ----------------------------------------------------------------------
def plot_thumbnail(t_peaks, f_peaks):
    fig, ax = plt.subplots(figsize=(2.4, 1.5))
    if len(t_peaks) > 800:
        idx = np.random.choice(len(t_peaks), size=800, replace=False)
        t_peaks, f_peaks = t_peaks[idx], f_peaks[idx]
    ax.scatter(t_peaks, f_peaks, s=1, c=ACCENT)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig


def plot_step1(result):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.3))
    ax1.pcolormesh(result["times"], result["frequencies"],
                    10 * np.log10(result["Sxx"] + 1e-10), cmap="magma", shading="auto")
    ax1.scatter(result["t_peaks"],result["f_peaks"],s=4,c='white',alpha=0.5)
    ax1.set_title("Spectrogram with Fingerprint Peaks", fontsize=10)
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("frequency (Hz)")

    ax2.scatter(result["t_peaks"], result["f_peaks"], s=4, c=ACCENT)
    ax2.set_title(f"Constellation ({len(result['t_peaks'])} peaks)", fontsize=10)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("frequency (Hz)")
    plt.tight_layout()
    return fig


def plot_step2(result, database):
    song = result["song"] or result["raw_best_song"]
    if song is None or song not in database["songs"]:
        return None
    meta = database["songs"][song]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.scatter(meta["t_peaks"], meta["f_peaks"], s=3, c=ACCENT, alpha=0.6)

    offset = result["best_offset"] or 0
    duration = result["query_duration"]
    ax.axvspan(offset, offset + duration, color=ACCENT_ORANGE, alpha=0.15)
    ax.axvline(offset, color=ACCENT_ORANGE, linewidth=1)
    ax.axvline(offset + duration, color=ACCENT_ORANGE, linewidth=1)

    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"Full fingerprint of '{os.path.splitext(song)[0]}' — query window highlighted",
                 fontsize=10)
    plt.tight_layout()
    return fig


def plot_step3(result):
    hist = result["best_hist"]
    if not hist:
        return None
    items = sorted(hist.items())
    xs = [o for o, c in items]
    ys = [c for o, c in items]
    peak_idx = int(np.argmax(ys))
    colors = [ACCENT_ORANGE if i == peak_idx else "#374151" for i in range(len(ys))]

    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.bar(xs, ys, width=0.5, color=colors)
    span = (max(xs) - min(xs)) if len(xs) > 1 else 1
    ax.annotate(
        f"{ys[peak_idx]} hashes\nalign here",
        xy=(xs[peak_idx], ys[peak_idx]),
        xytext=(xs[peak_idx] + span * 0.15, ys[peak_idx] * 0.85),
        color=ACCENT_ORANGE, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=ACCENT_ORANGE),
    )
    ax.set_xlabel("time offset (database time − query time), s")
    ax.set_ylabel("# hashes")
    ax.set_title("Step 3 · The alignment spike", fontsize=10)
    plt.tight_layout()
    return fig


def render_candidate_bars(candidates):
    if not candidates:
        return "<p style='color:#6b7280;'>No candidates matched.</p>"
    max_score = max(c[1] for c in candidates) or 1
    rows = ""
    for name, score, offset in candidates:
        pct = int(100 * score / max_score)
        display_name = os.path.splitext(name)[0]
        rows += f"""
        <div style="display:flex;align-items:center;margin-bottom:6px;">
          <div style="width:240px;font-size:13px;color:#e5e7eb;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{display_name}</div>
          <div style="flex:1;background:{PANEL};border-radius:4px;height:14px;margin:0 10px;overflow:hidden;">
            <div style="width:{pct}%;background:{ACCENT};height:100%;"></div>
          </div>
          <div style="width:50px;text-align:right;font-size:12px;color:#9ca3af;">{score}</div>
        </div>"""
    return rows


def render_match_banner(result):
    if result["song"]:
        name = os.path.splitext(result["song"])[0]
        ratio_text = f"{result['ratio']:.0f}x the runner-up" if result["ratio"] else "no runner-up"
        st.markdown(f"""
        <div style="border:1px solid {ACCENT}55;background:#0d1f1a;border-radius:8px;padding:18px 22px;margin:12px 0;">
          <div style="color:{ACCENT};font-size:11px;letter-spacing:2px;">MATCH FOUND</div>
          <div style="font-size:30px;font-weight:700;color:#fff;">{name}</div>
          <div style="color:#9ca3af;font-size:13px;">cluster score <span style="color:#fbbf24;">{result['score']}</span> · <span style="color:#fbbf24;">{ratio_text}</span></div>
        </div>""", unsafe_allow_html=True)
    else:
        guess = os.path.splitext(result["raw_best_song"])[0] if result["raw_best_song"] else "—"
        st.markdown(f"""
        <div style="border:1px solid #ef444455;background:#1f0d0d;border-radius:8px;padding:18px 22px;margin:12px 0;">
          <div style="color:#ef4444;font-size:11px;letter-spacing:2px;">NO MATCH</div>
          <div style="font-size:22px;font-weight:700;color:#fff;">No candidate cleared the confidence threshold</div>
          <div style="color:#9ca3af;font-size:13px;">best guess: {guess} · score {result['score']}</div>
        </div>""", unsafe_allow_html=True)


def render_pipeline_timing(timings, best_offset):
    cols = st.columns(5)
    steps = [
        ("①", "SPECTROGRAM", f"{timings['spectrogram_ms']:.0f} ms", timings["spectrogram_shape"]),
        ("②", "CONSTELLATION", f"{timings['constellation_ms']:.0f} ms", f"{timings['n_peaks']} peaks"),
        ("③", "HASHING", f"{timings['hashing_ms']:.0f} ms", f"{timings['n_hashes']:,} hashes"),
        ("④", "DB LOOKUP", f"{timings['db_lookup_ms']:.0f} ms", f"{timings['n_tracks_in_db']} tracks"),
        ("⑤", "SCORING", f"{timings['scoring_ms']:.0f} ms", f"offset {best_offset}"),
    ]
    for col, (index, label, value, sub) in zip(cols, steps):
        col.markdown(f"""
        <div style="text-align:center;">
          <div style="font-size:10px;color:{ACCENT};letter-spacing:1px;">{index}</div>
          <div style="font-size:10px;color:{MUTED};letter-spacing:1px;">{label}</div>
          <div style="font-size:17px;color:{WHITE};font-weight:700;">{value}</div>
          <div style="font-size:10px;color:{MUTED};">{sub}</div>
        </div>""", unsafe_allow_html=True)
    st.caption(f"total {timings['total_ms']:.0f} ms")


def run_identification(data, sr):
    result = identify_song(data, sr, database)

    render_pipeline_timing(result["timings"], result["best_offset"])
    render_match_banner(result)

    st.markdown("**Candidate scores**")
    st.markdown(render_candidate_bars(result["candidates"]), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <div style="color: {ACCENT}; font-size: 10px; font-weight: 600; 
                    letter-spacing: 1px; text-transform: uppercase;">
            STEP 1 · FEATURE EXTRACTION
        </div>
        <h2 style="color: white; margin-top: 4px; font-size: 20px;">
            From spectrogram to constellation
        </h2>
        <p style="color: {MUTED}; font-size: 15px; line-height: 1.6;">
            The clip was converted into a time-frequency map (left); brighter means louder at that frequency and moment. From that rich image, only the 
              <b style="color: {ACCENT};">{len(result['t_peaks']):,} most prominent peaks</b> were kept (right). Discarding amplitude and phase makes the
            robust to EQ, volume changes, and mild noise.
        </p>
    </div>
  """, unsafe_allow_html=True)
    st.pyplot(plot_step1(result), clear_figure=True)

    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <div style="color: {ACCENT}; font-size: 10px; font-weight: 600; 
                    letter-spacing: 1px; text-transform: uppercase;">
            Step 2 · DATABASE SEARCH
        </div>
        <h2 style="color: white; margin-top: 4px; font-size: 20px;">
            Where in the song?
        </h2>
        <p style="color: {MUTED}; font-size: 15px; line-height: 1.6;">
            The <b style="color: oasis;"> fingerprint hashes<\b> were looked up against every indexed track. Below is the full fingerprint of <i> <\i>
            reconstructed from the database. Each dot is a stored hash anchor. The highlighted window is exactly where yhe query clip sits inside the full song.
        </p>
    </div>
  """, unsafe_allow_html=True)
    fig2 = plot_step2(result, database)
    if fig2 is not None:
        st.pyplot(fig2, clear_figure=True)
    else:
        st.caption("No matching track to reconstruct against.")

    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <div style="color: {ACCENT}; font-size: 10px; font-weight: 600; 
                    letter-spacing: 1px; text-transform: uppercase;">
            STEP 3 · THE PROOF
        </div>
        <h2 style="color: white; margin-top: 4px; font-size: 20px;">
            The alignment spike
        </h2>
        <p style="color: {MUTED}; font-size: 15px; line-height: 1.6;">
            Every matched hash votes for a time offset (database frame minus query frame). Chance matches scatter votes randomly, forming a flat noise floor. A genuine match makes them 
            converge: <b style="color: {ACCENT_ORANGE};"> agreed on a single offset<\b>. That spike cannot be a coincidence.
        </p>
    </div>
  """, unsafe_allow_html=True)
    fig3 = plot_step3(result)
    if fig3 is not None:
        st.pyplot(fig3, clear_figure=True)
    else:
        st.caption("No matching hashes found against any song in the database.")


# ----------------------------------------------------------------------
# LIBRARY TAB
# ----------------------------------------------------------------------
with library_tab:
    st.caption("LIBRARY")
    st.subheader("In the database")
    st.markdown(
        f"<div style='color:{MUTED};font-size:13px;margin-bottom:14px;'>"
        f"Song indexing is managed by the admin. Drop a clip in the Identify tab to test the library."
        f"</div>", unsafe_allow_html=True
    )

    songs = list(database["songs"].items())
    cols_per_row = 4
    for i in range(0, len(songs), cols_per_row):
        row = songs[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (song_name, meta) in zip(cols, row):
            with col:
                st.pyplot(plot_thumbnail(meta["t_peaks"], meta["f_peaks"]), clear_figure=True)
                st.markdown(f"**{os.path.splitext(song_name)[0]}**")
                st.caption(f"{meta['n_hashes']:,} hashes")


# ----------------------------------------------------------------------
# IDENTIFY TAB
# ----------------------------------------------------------------------
with identify_tab:
    st.caption("SEARCH")
    st.subheader("Identify a clip")

    uploaded = st.file_uploader(
        "Upload a query clip", type=[e.strip(".") for e in AUDIO_EXTENSIONS],
        key="single_upload", label_visibility="collapsed"
    )
    st.caption("200MB per file · WAV, MP3, FLAC, OGG, M4A")

    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = None

    if os.path.isdir(SAMPLES_DIR):
        sample_files = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(AUDIO_EXTENSIONS))
        if sample_files:
            st.markdown("**OR TRY A SAMPLE**")
            for sf in sample_files:
                c1, c2 = st.columns([6, 1])
                with c1:
                    st.audio(os.path.join(SAMPLES_DIR, sf))
                with c2:
                    if st.button("Try", key=f"try_{sf}"):
                        st.session_state.selected_sample = sf

    run = st.button("Identify", type="primary")

    query_source, query_name = None, None
    if uploaded is not None:
        query_source, query_name = uploaded, uploaded.name
    elif st.session_state.selected_sample:
        query_name = st.session_state.selected_sample
        query_source = os.path.join(SAMPLES_DIR, query_name)

    if run:
        if query_source is None:
            st.warning("Upload a clip or pick a sample first.")
        else:
            with st.spinner("Fingerprinting..."):
                data, sr = load_audio(query_source, filename=query_name)
            run_identification(data, sr)


# ----------------------------------------------------------------------
# BATCH TAB
# ----------------------------------------------------------------------
with batch_tab:
    st.caption("BATCH")
    st.subheader("Identify many clips at once")
    st.markdown(
        f"<div style='color:{MUTED};font-size:13px;margin-bottom:14px;'>"
        f"Upload a set of query clips. Each is identified against the currently indexed "
        f"library, and the results are written to a standardised <code>results.csv</code> "
        f"with columns <code>filename, prediction</code>. The <code>prediction</code> is the "
        f"matched track's filename without extension, or <code>none</code> when no candidate "
        f"clears the confidence threshold.</div>", unsafe_allow_html=True
    )

    batch_files = st.file_uploader(
        "Upload clips", type=[e.strip(".") for e in AUDIO_EXTENSIONS],
        accept_multiple_files=True, key="batch_upload", label_visibility="collapsed"
    )
    st.caption("200MB per file · WAV, MP3, FLAC, OGG, M4A")

    if batch_files and st.button("Run batch", type="primary"):
        rows = []
        progress = st.progress(0, text=f"Identifying ... 0/{len(batch_files)}")

        for i, f in enumerate(batch_files):
            data, sr = load_audio(f, filename=f.name)
            result = identify_song(data, sr, database)
            prediction = os.path.splitext(result["song"])[0] if result["song"] else "none"
            rows.append({"filename": f.name, "prediction": prediction})
            progress.progress((i + 1) / len(batch_files), text=f"Identifying ... {i + 1}/{len(batch_files)}")

        results_df = pd.DataFrame(rows)
        st.markdown("**Results**")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        n_matched = int((results_df["prediction"] != "none").sum())
        n_none = int((results_df["prediction"] == "none").sum())
        st.caption(f"{n_matched} / {len(results_df)} clips matched to a track ({n_none} returned none).")

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results.csv", data=csv_bytes,
                            file_name="results.csv", mime="text/csv")
