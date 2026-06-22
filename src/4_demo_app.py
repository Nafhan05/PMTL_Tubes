"""
4_demo_app.py — Aplikasi Demo Interaktif Deteksi Wireless Jamming.

Jalankan: streamlit run src/4_demo_app.py

Features:
- Pilih model (1D-CNN / LSTM)
- Pilih jenis jamming (None / CW / Barrage)
- Atur parameter SJR, efek dunia nyata
- Visualisasi sinyal + animasi network
- Prediksi real-time dari model terlatih
"""

import os
import sys
import warnings

# Suppress st.cache deprecation warnings (Streamlit 1.22 + TF 2.10 protobuf conflict)
warnings.filterwarnings("ignore", message=".*st\.cache.*deprecated.*")
warnings.filterwarnings("ignore", message=".*CacheFuncHasher.*")

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import h5py
import time

# Import komponen lokal
from src.demo_components import (
    apply_doppler_shift,
    apply_multipath_fading,
    create_network_animation_html,
    run_inference,
)
from src.data_loader import HDF5_FILE, LABEL_NORMAL, LABEL_CW, LABEL_BARRAGE
from src.data_loader import inject_cw_jamming, inject_barrage_jamming

# --------------------------------------------------------------------------- #
#                          PAGE CONFIG                                          #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Wireless Jamming Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 8px 0 16px;
    }
    .main-header h1 {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 14px;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .section-title {
        font-size: 16px;
        font-weight: 600;
        color: #e2e8f0;
        margin: 20px 0 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#                          LOAD MODEL (CACHED)                                  #
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model(model_name: str):
    """Load TensorFlow model (cached agar tidak reload setiap interaksi)."""
    # GPU setup
    try:
        import src.gpu_setup
    except Exception:
        pass

    import tensorflow as tf

    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    if model_name == "1D-CNN":
        path = os.path.join(models_dir, "best_1dcnn.keras")
    elif model_name == "1D-CNN (HPO)":
        path = os.path.join(models_dir, "best_1dcnn_hpo.keras")
    elif model_name == "LSTM":
        path = os.path.join(models_dir, "best_lstm.keras")
    elif model_name == "2D-CNN":
        path = os.path.join(models_dir, "best_2dcnn.keras")
    elif model_name == "2D-CNN (HPO)":
        path = os.path.join(models_dir, "best_2dcnn_hpo.keras")
    else:
        return None

    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


@st.cache_data(show_spinner=False)
def load_signal(idx: int):
    """Load satu sinyal dari dataset HDF5."""
    if not os.path.exists(HDF5_FILE):
        return None
    with h5py.File(HDF5_FILE, "r") as f:
        total = f["X"].shape[0]
        idx = min(idx, total - 1)
        signal = f["X"][idx][:]  # copy to regular numpy
    return signal.astype(np.float32).tolist()  # serialize as list for caching


@st.cache_data(show_spinner=False)
def get_dataset_size():
    """Dapatkan jumlah total frame dalam dataset."""
    if not os.path.exists(HDF5_FILE):
        return 0
    with h5py.File(HDF5_FILE, "r") as f:
        return f["X"].shape[0]


# --------------------------------------------------------------------------- #
#                              SIDEBAR                                          #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    # Model selection
    st.markdown("### Model")
    model_name = st.selectbox(
        "Pilih Model",
        ["1D-CNN", "1D-CNN (HPO)", "LSTM", "2D-CNN", "2D-CNN (HPO)"],
        help="1D-CNN HPO sangat optimal, LSTM akurat, 2D-CNN menggunakan spektrogram",
    )

    st.markdown("---")

    # Jamming configuration
    st.markdown("### Jamming Configuration")
    jamming_type = st.selectbox(
        "Jenis Jamming",
        ["None (Normal)", "CW Jamming", "Barrage Jamming"],
        help="CW = gelombang sinus konstan, Barrage = noise wideband",
    )

    sjr_db = st.slider(
        "Signal-to-Jamming Ratio (dB)",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        help="Semakin rendah = jamming semakin kuat. 0 dB = power sama.",
        disabled=jamming_type == "None (Normal)",
    )

    st.markdown("---")

    # Real-world effects
    st.markdown("### Efek Dunia Nyata")

    enable_doppler = st.checkbox("Doppler Shift", help="Simulasi pergerakan jammer")
    if enable_doppler:
        doppler_speed = st.slider("Kecepatan (km/h)", 10, 200, 60)
    else:
        doppler_speed = 0

    enable_multipath = st.checkbox("Multipath Fading", help="Sinyal memantul dari berbagai objek")
    if enable_multipath:
        n_paths = st.slider("Jumlah Path", 2, 6, 3)
    else:
        n_paths = 1

    st.markdown("---")

    # Signal source
    st.markdown("### Source Signal")
    ds_size = get_dataset_size()
    if ds_size > 0:
        signal_idx = st.number_input(
            f"Index Sinyal (0 - {ds_size - 1})",
            min_value=0,
            max_value=ds_size - 1,
            value=42,
        )
    else:
        signal_idx = 0
        st.warning("Dataset tidak ditemukan!")

    # Run button
    st.markdown("---")
    run_btn = st.button("🚀 Jalankan Deteksi", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
#                           MAIN CONTENT                                        #
# --------------------------------------------------------------------------- #
st.markdown("""
<div class="main-header">
    <h1>Wireless Jamming Detection System</h1>
    <p>Simulasi & Deteksi Jamming Berbasis Deep Learning</p>
</div>
""", unsafe_allow_html=True)


# Determine jamming type key
jam_key = "none"
if "CW" in jamming_type:
    jam_key = "cw"
elif "Barrage" in jamming_type:
    jam_key = "barrage"


# --------------------------------------------------------------------------- #
#                 PROCESS SIGNAL & RUN DETECTION                                #
# --------------------------------------------------------------------------- #
if run_btn and ds_size > 0:
    # Load original signal (cached as list, convert to numpy)
    cached_signal = load_signal(signal_idx)

    if cached_signal is None:
        st.error("Gagal memuat sinyal dari dataset!")
    else:
        original_signal = np.array(cached_signal, dtype=np.float32)

        # Apply jamming
        jammed_signal = original_signal.copy()
        actual_label = "Normal"

        if jam_key == "cw":
            jammed_signal = inject_cw_jamming(jammed_signal, sjr_db)
            actual_label = "CW Jamming"
        elif jam_key == "barrage":
            jammed_signal = inject_barrage_jamming(jammed_signal, sjr_db)
            actual_label = "Barrage Jamming"

        # Apply real-world effects
        processed_signal = jammed_signal.copy()
        effects_applied = []

        if enable_doppler:
            processed_signal = apply_doppler_shift(processed_signal, speed_kmh=doppler_speed)
            effects_applied.append(f"Doppler ({doppler_speed} km/h)")

        if enable_multipath:
            processed_signal = apply_multipath_fading(processed_signal, n_paths=n_paths)
            effects_applied.append(f"Multipath ({n_paths} paths)")

        # Store in session state as plain numpy arrays
        st.session_state["original"] = np.array(original_signal, dtype=np.float32)
        st.session_state["processed"] = np.array(processed_signal, dtype=np.float32)
        st.session_state["actual_label"] = actual_label
        st.session_state["effects"] = effects_applied
        st.session_state["jam_key"] = jam_key
        st.session_state["sjr_db"] = sjr_db

        # Run inference
        model = load_model(model_name)
        if model is not None:
            model_type = "2d" if "2D-CNN" in model_name else "1d"
            seq_len = 1024
            
            # Khusus LSTM atau 1D-CNN HPO mungkin butuh sequence lebih pendek
            if model_type == "1d" and len(model.input_shape) >= 2 and model.input_shape[1] is not None and model.input_shape[1] != 1024:
                seq_len = model.input_shape[1]
                
            nperseg = 64
            hop_length = 16
            
            # Coba load HPO config jika modelnya 2D-CNN HPO
            if model_name == "2D-CNN (HPO)":
                hpo_json = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "hpo_best_2dcnn.json")
                if os.path.exists(hpo_json):
                    import json
                    with open(hpo_json) as f:
                        cfg = json.load(f)
                    nperseg = cfg.get("stft_nperseg", 64)
                    hop_length = cfg.get("stft_hop_length", 16)
            
            result = run_inference(
                model, 
                processed_signal, 
                seq_len=seq_len,
                model_type=model_type,
                nperseg=nperseg,
                hop_length=hop_length
            )
            st.session_state["result"] = result
            st.session_state["model_name"] = model_name
        else:
            st.error(f"Model {model_name} tidak ditemukan di folder models/!")
            st.session_state["result"] = None

elif "result" not in st.session_state:
    # Default state — no result yet
    st.session_state.setdefault("jam_key", "none")
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("sjr_db", 0.0)


# --------------------------------------------------------------------------- #
#                    NETWORK ANIMATION                                          #
# --------------------------------------------------------------------------- #
result = st.session_state.get("result")
current_jam_key = st.session_state.get("jam_key", "none")
current_sjr = st.session_state.get("sjr_db", 0.0)

if result:
    prediction = result["prediction"]
    confidence = result["confidence"]
else:
    prediction = "Menunggu..."
    confidence = 0.0

st.markdown('<div class="section-title">Network Diagram</div>', unsafe_allow_html=True)
animation_html = create_network_animation_html(
    jamming_type=current_jam_key,
    prediction=prediction,
    confidence=confidence,
    sjr_db=current_sjr,
)
st.components.v1.html(animation_html, height=280)


# --------------------------------------------------------------------------- #
#                    DETECTION RESULTS                                          #
# --------------------------------------------------------------------------- #
if result:
    st.markdown('<div class="section-title">Hasil Deteksi</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    pred_color_map = {"Normal": "#22c55e", "CW Jamming": "#ef4444", "Barrage Jamming": "#f97316"}
    pred_color = pred_color_map.get(result["prediction"], "#666")

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color: {pred_color}">{result['prediction']}</div>
            <div class="label">Prediksi Model</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color: #38bdf8">{result['confidence']*100:.1f}%</div>
            <div class="label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        actual = st.session_state.get("actual_label", "Normal")
        actual_color = pred_color_map.get(actual, "#666")
        match = "✓" if result["prediction"] == actual else "✗"
        match_color = "#22c55e" if result["prediction"] == actual else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color: {actual_color}">{actual} <span style="color:{match_color}">{match}</span></div>
            <div class="label">Label Aktual</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value" style="color: #a78bfa">{result['latency_ms']:.1f} ms</div>
            <div class="label">Inference Latency</div>
        </div>
        """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("")
    prob_cols = st.columns(3)
    for i, (name, prob) in enumerate(result["probabilities"].items()):
        with prob_cols[i]:
            color = pred_color_map.get(name, "#666")
            st.markdown(f"**{name}**: {prob*100:.1f}%")
            st.progress(min(prob, 1.0))

    # Effects info
    effects = st.session_state.get("effects", [])
    if effects:
        st.info(f"Efek diterapkan: {', '.join(effects)}")


# --------------------------------------------------------------------------- #
#                    SIGNAL VISUALIZATION                                       #
# --------------------------------------------------------------------------- #
if "original" in st.session_state and "processed" in st.session_state:
    st.markdown('<div class="section-title">Visualisasi Sinyal</div>', unsafe_allow_html=True)

    original = np.array(st.session_state["original"], dtype=np.float32)
    processed = np.array(st.session_state["processed"], dtype=np.float32)

    tab1, tab2, tab3 = st.tabs(["📈 Time Domain", "🔵 Konstelasi", "📊 Power Spectrum"])

    with tab1:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("In-Phase (I)", "Quadrature (Q)"),
            vertical_spacing=0.12,
        )

        orig_i = original[:, 0].tolist()
        orig_q = original[:, 1].tolist()
        proc_i = processed[:, 0].tolist()
        proc_q = processed[:, 1].tolist()
        x_axis = list(range(len(orig_i)))

        # Original signal
        fig.add_trace(
            go.Scatter(x=x_axis, y=orig_i, name="Original I",
                       line=dict(color="#3b82f6", width=1), opacity=0.5),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_axis, y=orig_q, name="Original Q",
                       line=dict(color="#3b82f6", width=1), opacity=0.5),
            row=2, col=1,
        )

        # Processed signal
        fig.add_trace(
            go.Scatter(x=x_axis, y=proc_i, name="Processed I",
                       line=dict(color="#ef4444", width=1)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x_axis, y=proc_q, name="Processed Q",
                       line=dict(color="#f97316", width=1)),
            row=2, col=1,
        )

        fig.update_layout(
            height=450,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,1)",
            font=dict(family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.08),
            margin=dict(t=40, b=20, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=original[:, 0].tolist(), y=original[:, 1].tolist(),
            mode="markers", name="Original",
            marker=dict(size=2, color="#3b82f6", opacity=0.3),
        ))

        fig.add_trace(go.Scatter(
            x=processed[:, 0].tolist(), y=processed[:, 1].tolist(),
            mode="markers", name="Processed",
            marker=dict(size=2, color="#ef4444", opacity=0.5),
        ))

        fig.update_layout(
            height=450,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,1)",
            xaxis_title="In-Phase (I)",
            yaxis_title="Quadrature (Q)",
            font=dict(family="Inter"),
            margin=dict(t=20, b=40, l=40, r=20),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Power Spectral Density
        complex_orig = original[:, 0] + 1j * original[:, 1]
        complex_proc = processed[:, 0] + 1j * processed[:, 1]

        fft_orig = np.fft.fftshift(np.abs(np.fft.fft(complex_orig)))
        fft_proc = np.fft.fftshift(np.abs(np.fft.fft(complex_proc)))

        psd_orig = (20 * np.log10(fft_orig + 1e-10)).tolist()
        psd_proc = (20 * np.log10(fft_proc + 1e-10)).tolist()
        freqs = np.linspace(-0.5, 0.5, len(psd_orig)).tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs, y=psd_orig, name="Original",
            line=dict(color="#3b82f6", width=1), opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=freqs, y=psd_proc, name="Processed",
            line=dict(color="#ef4444", width=1.5),
        ))

        fig.update_layout(
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,1)",
            xaxis_title="Normalized Frequency",
            yaxis_title="Power (dB)",
            font=dict(family="Inter"),
            margin=dict(t=20, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # Placeholder
    st.markdown("")
    st.markdown(
        '<p style="text-align:center; color: #64748b; padding: 40px 0;">'
        'Klik <b>"Jalankan Deteksi"</b> di sidebar untuk memulai simulasi</p>',
        unsafe_allow_html=True,
    )
