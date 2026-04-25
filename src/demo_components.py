"""
demo_components.py — Komponen helper untuk demo interaktif deteksi jamming.

Berisi:
- Fungsi efek dunia nyata (Doppler, multipath, intermittent)
- HTML/CSS/JS animasi network diagram
- Fungsi inference model
"""

import numpy as np
import time


# --------------------------------------------------------------------------- #
#                    EFEK DUNIA NYATA (REAL-WORLD EFFECTS)                      #
# --------------------------------------------------------------------------- #
def apply_doppler_shift(signal: np.ndarray, speed_kmh: float = 50.0, carrier_freq_mhz: float = 900.0) -> np.ndarray:
    """
    Simulasi Doppler shift akibat pergerakan jammer/receiver.
    Menggeser frekuensi sinyal berdasarkan kecepatan relatif.
    """
    c = 3e8  # speed of light (m/s)
    v = speed_kmh * 1000 / 3600  # convert to m/s
    fc = carrier_freq_mhz * 1e6  # convert to Hz
    fd = (v / c) * fc  # Doppler frequency shift

    n_samples = signal.shape[0]
    t = np.arange(n_samples) / 1e6  # normalized time
    phase_shift = 2 * np.pi * fd * t

    # Apply phase rotation to I/Q
    i_shifted = signal[:, 0] * np.cos(phase_shift) - signal[:, 1] * np.sin(phase_shift)
    q_shifted = signal[:, 0] * np.sin(phase_shift) + signal[:, 1] * np.cos(phase_shift)

    return np.stack([i_shifted, q_shifted], axis=-1)


def apply_multipath_fading(signal: np.ndarray, n_paths: int = 3, max_delay: int = 10) -> np.ndarray:
    """
    Simulasi multipath fading — sinyal datang dari beberapa jalur berbeda
    dengan delay dan atenuasi yang berbeda (Rayleigh fading sederhana).
    """
    result = np.zeros_like(signal, dtype=np.float64)
    rng = np.random.RandomState(42)

    for _ in range(n_paths):
        delay = rng.randint(0, max_delay + 1)
        attenuation = rng.uniform(0.3, 1.0)
        phase = rng.uniform(0, 2 * np.pi)

        # Delay the signal
        delayed = np.zeros_like(signal)
        if delay > 0:
            delayed[delay:] = signal[:-delay]
        else:
            delayed = signal.copy()

        # Apply attenuation and phase rotation
        i_rot = delayed[:, 0] * np.cos(phase) - delayed[:, 1] * np.sin(phase)
        q_rot = delayed[:, 0] * np.sin(phase) + delayed[:, 1] * np.cos(phase)

        result[:, 0] += attenuation * i_rot
        result[:, 1] += attenuation * q_rot

    # Normalize to original power
    orig_power = np.mean(signal**2) + 1e-10
    result_power = np.mean(result**2) + 1e-10
    result *= np.sqrt(orig_power / result_power)

    return result


def apply_intermittent_jamming(signal: np.ndarray, y_label: int, period_samples: int = 256) -> tuple:
    """
    Simulasi intermittent jamming — jammer nyala-mati secara periodik.
    Mengembalikan sinyal yang hanya terjam sebagian.
    """
    n = signal.shape[0]
    mask = np.zeros(n, dtype=bool)

    # On/off pattern
    for start in range(0, n, period_samples * 2):
        end = min(start + period_samples, n)
        mask[start:end] = True

    # Only keep jamming in 'on' regions, use original signal in 'off' regions
    # Since we already have the jammed signal, we partially revert it
    return signal, mask


# --------------------------------------------------------------------------- #
#                      MODEL INFERENCE                                          #
# --------------------------------------------------------------------------- #
def run_inference(model, signal: np.ndarray, seq_len: int = 1024) -> dict:
    """
    Jalankan inference pada satu sinyal.

    Returns:
        dict with keys: prediction, confidence, probabilities, latency_ms
    """
    CLASS_NAMES = ["Normal", "CW Jamming", "Barrage Jamming"]

    # Prepare input
    x = signal.copy()
    if seq_len < x.shape[0]:
        indices = np.linspace(0, x.shape[0] - 1, seq_len, dtype=int)
        x = x[indices]

    x = x.reshape(1, seq_len, 2).astype(np.float32)

    # Warm-up
    model.predict(x, verbose=0)

    # Timed prediction
    start = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    elapsed = (time.perf_counter() - start) * 1000

    pred_idx = int(np.argmax(probs))

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "pred_idx": pred_idx,
        "confidence": float(probs[pred_idx]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        "latency_ms": elapsed,
    }


# --------------------------------------------------------------------------- #
#                     NETWORK ANIMATION HTML                                    #
# --------------------------------------------------------------------------- #
def create_network_animation_html(
    jamming_type: str = "none",
    prediction: str = "Normal",
    confidence: float = 0.0,
    sjr_db: float = 0.0,
) -> str:
    """
    Generate HTML/CSS/JS for the animated network diagram.

    Layout (horizontal flow):
        [TX] ----signal----> [RX] ----data----> [AI MODEL]
                              ^                     |
                              |                     v
                          [JAMMER]             [PREDICTION]
    """

    jammer_visible = "visible" if jamming_type != "none" else "hidden"
    jam_label = "CW Jammer" if jamming_type == "cw" else "Barrage Jammer" if jamming_type == "barrage" else ""
    jam_color = "#ef4444" if jamming_type == "cw" else "#f97316" if jamming_type == "barrage" else "#666"

    # Prediction colors
    pred_colors = {"Normal": "#22c55e", "CW Jamming": "#ef4444", "Barrage Jamming": "#f97316"}
    pred_color = pred_colors.get(prediction, "#94a3b8")
    conf_pct = f"{confidence * 100:.1f}%"

    jam_desc = f"SJR: {sjr_db:.0f} dB" if jamming_type != "none" else ""

    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        .net-container {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 16px;
            padding: 24px;
            position: relative;
            overflow: hidden;
            height: 260px;
            border: 1px solid rgba(255,255,255,0.08);
        }}

        .net-container::before {{
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 30% 60%, rgba(56,189,248,0.04) 0%, transparent 50%);
        }}

        /* Grid background */
        .grid-bg {{
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
            background-size: 40px 40px;
        }}

        /* ---- NODES ---- */
        .node {{
            position: absolute;
            text-align: center;
            z-index: 10;
        }}

        .node-icon {{
            width: 64px;
            height: 64px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            margin: 0 auto 6px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.35);
        }}

        .node-label {{
            color: #cbd5e1;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
        }}

        /* Transmitter — far left, vertically centered */
        .transmitter {{
            left: 5%;
            top: 50%;
            transform: translateY(-50%);
        }}
        .transmitter .node-icon {{
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            border: 2px solid #60a5fa;
        }}

        /* Receiver — center, vertically centered */
        .receiver {{
            left: 38%;
            top: 50%;
            transform: translate(-50%, -50%);
        }}
        .receiver .node-icon {{
            background: linear-gradient(135deg, #22c55e, #16a34a);
            border: 2px solid #4ade80;
        }}

        /* AI Model — right side */
        .model-node {{
            right: 16%;
            top: 50%;
            transform: translateY(-50%);
        }}
        .model-node .node-icon {{
            background: linear-gradient(135deg, #8b5cf6, #7c3aed);
            border: 2px solid #a78bfa;
        }}

        /* Jammer — above receiver */
        .jammer {{
            left: 38%;
            top: 0px;
            transform: translateX(-50%);
            visibility: {jammer_visible};
        }}
        .jammer .node-icon {{
            background: linear-gradient(135deg, {jam_color}, #991b1b);
            border: 2px solid {jam_color};
            animation: pulse 2s ease-in-out infinite;
            width: 54px;
            height: 54px;
            font-size: 24px;
        }}

        .jam-info {{
            position: absolute;
            left: 50%;
            top: 60px;
            transform: translateX(-50%);
            color: {jam_color};
            font-size: 10px;
            font-weight: 600;
            visibility: {jammer_visible};
            background: rgba(0,0,0,0.5);
            padding: 2px 8px;
            border-radius: 4px;
            white-space: nowrap;
        }}

        /* ---- CONNECTION LINES ---- */

        /* TX → RX */
        .line-tx-rx {{
            position: absolute;
            top: 50%;
            left: 14%;
            width: 20%;
            height: 1px;
            background: linear-gradient(90deg, rgba(56,189,248,0.4), rgba(34,197,94,0.4));
        }}

        /* RX → Model */
        .line-rx-model {{
            position: absolute;
            top: 50%;
            left: 44%;
            width: 24%;
            height: 1px;
            background: linear-gradient(90deg, rgba(34,197,94,0.3), rgba(139,92,246,0.4));
        }}

        /* Jammer → RX (vertical) */
        .line-jam {{
            position: absolute;
            top: 58px;
            left: 38%;
            width: 1px;
            height: calc(50% - 58px);
            background: linear-gradient(180deg, {jam_color}55, transparent);
            visibility: {jammer_visible};
        }}

        /* ---- SIGNAL WAVES TX→RX ---- */
        .signal-path {{
            position: absolute;
            top: 50%;
            left: 14%;
            width: 20%;
            height: 2px;
            transform: translateY(-50%);
        }}

        .signal-wave {{
            position: absolute;
            width: 28px;
            height: 28px;
            top: -13px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(56,189,248,0.6) 0%, transparent 70%);
            animation: moveSignal 2.5s linear infinite;
        }}
        .signal-wave:nth-child(2) {{ animation-delay: -0.8s; }}
        .signal-wave:nth-child(3) {{ animation-delay: -1.6s; }}

        @keyframes moveSignal {{
            0% {{ left: -5%; opacity: 0; }}
            10% {{ opacity: 1; }}
            85% {{ opacity: 1; }}
            100% {{ left: 105%; opacity: 0; }}
        }}

        /* ---- DATA WAVES RX→MODEL ---- */
        .data-path {{
            position: absolute;
            top: 50%;
            left: 44%;
            width: 24%;
            height: 2px;
            transform: translateY(-50%);
        }}

        .data-wave {{
            position: absolute;
            width: 20px;
            height: 20px;
            top: -9px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(139,92,246,0.5) 0%, transparent 70%);
            animation: moveData 2s linear infinite;
        }}
        .data-wave:nth-child(2) {{ animation-delay: -0.7s; }}

        @keyframes moveData {{
            0% {{ left: -5%; opacity: 0; }}
            15% {{ opacity: 1; }}
            80% {{ opacity: 1; }}
            100% {{ left: 105%; opacity: 0; }}
        }}

        /* ---- JAM WAVES ---- */
        .jam-path {{
            position: absolute;
            top: 58px;
            left: 38%;
            width: 2px;
            height: calc(50% - 58px);
            transform: translateX(-50%);
            visibility: {jammer_visible};
        }}

        .jam-wave {{
            position: absolute;
            width: 24px;
            height: 24px;
            left: -11px;
            border-radius: 50%;
            background: radial-gradient(circle, {jam_color}aa 0%, transparent 70%);
            animation: moveJam 1.8s linear infinite;
        }}
        .jam-wave:nth-child(2) {{ animation-delay: -0.6s; }}
        .jam-wave:nth-child(3) {{ animation-delay: -1.2s; }}

        @keyframes moveJam {{
            0% {{ top: 0%; opacity: 0; }}
            15% {{ opacity: 1; }}
            80% {{ opacity: 0.7; }}
            100% {{ top: 100%; opacity: 0; }}
        }}

        @keyframes pulse {{
            0%, 100% {{ box-shadow: 0 0 8px {jam_color}44; }}
            50% {{ box-shadow: 0 0 24px {jam_color}88; }}
        }}

        /* ---- PREDICTION BADGE ---- */
        .pred-badge {{
            position: absolute;
            right: 3%;
            top: 50%;
            transform: translateY(-50%);
            background: {pred_color}18;
            border: 1px solid {pred_color}88;
            border-radius: 10px;
            padding: 10px 16px;
            z-index: 20;
            text-align: center;
            min-width: 100px;
        }}

        .pred-badge .pred-text {{
            color: {pred_color};
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 2px;
        }}

        .pred-badge .pred-conf {{
            color: #94a3b8;
            font-size: 11px;
        }}

        /* Arrow label */
        .arrow-label {{
            position: absolute;
            font-size: 9px;
            color: #475569;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .arrow-label.tx-rx {{
            top: calc(50% - 16px);
            left: 19%;
        }}
        .arrow-label.rx-model {{
            top: calc(50% - 16px);
            left: 52%;
        }}
    </style>

    <div class="net-container">
        <div class="grid-bg"></div>

        <!-- Arrow labels -->
        <div class="arrow-label tx-rx">RF SIGNAL</div>
        <div class="arrow-label rx-model">I/Q DATA</div>

        <!-- Connection lines -->
        <div class="line-tx-rx"></div>
        <div class="line-rx-model"></div>
        <div class="line-jam"></div>

        <!-- Transmitter -->
        <div class="node transmitter">
            <div class="node-icon">📡</div>
            <div class="node-label">Transmitter</div>
        </div>

        <!-- Receiver -->
        <div class="node receiver">
            <div class="node-icon">📶</div>
            <div class="node-label">Receiver</div>
        </div>

        <!-- AI Model -->
        <div class="node model-node">
            <div class="node-icon">🧠</div>
            <div class="node-label">AI Model</div>
        </div>

        <!-- Jammer -->
        <div class="node jammer">
            <div class="node-icon">⚡</div>
            <div class="node-label">{jam_label}</div>
        </div>
        <div class="jam-info">{jam_desc}</div>

        <!-- Signal waves TX → RX -->
        <div class="signal-path">
            <div class="signal-wave"></div>
            <div class="signal-wave"></div>
            <div class="signal-wave"></div>
        </div>

        <!-- Data waves RX → Model -->
        <div class="data-path">
            <div class="data-wave"></div>
            <div class="data-wave"></div>
        </div>

        <!-- Jamming waves -->
        <div class="jam-path">
            <div class="jam-wave"></div>
            <div class="jam-wave"></div>
            <div class="jam-wave"></div>
        </div>

        <!-- Prediction result -->
        <div class="pred-badge">
            <div class="pred-text">{prediction}</div>
            <div class="pred-conf">{conf_pct}</div>
        </div>
    </div>
    """
    return html
