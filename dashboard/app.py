"""
F1 Race Predictor — Streamlit Dashboard
Styled to match the official Formula 1 website aesthetic.

Run:
    streamlit run dashboard/app.py

Car images:
    Place the following files in dashboard/assets/
        car_red_bull.png   — Red Bull RB20
        car_ferrari.png    — Ferrari SF-24
        car_mercedes.png   — Mercedes W15
        car_mclaren.png    — McLaren MCL38
        f1_hero.png        — hero banner image (wide)
    Images are optional — the app renders gracefully without them.
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from dashboard.api_client import (
    get_health,
    predict_lap_time,
    predict_pit_stop,
    predict_position,
)
from dashboard.components import (
    lap_history_chart,
    lap_time_gauge,
    pit_probability_bar,
    position_podium,
    render_api_status,
)

ASSETS = Path(__file__).parent / "assets"


def asset(name: str) -> Path | None:
    """Return path if file exists, else None."""
    p = ASSETS / name
    return p if p.exists() else None


# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── F1 Global CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@400;500;600&display=swap');

:root {
    --f1-red:    #E8002D;
    --f1-dark:   #15151E;
    --f1-panel:  #1e1e2e;
    --f1-border: #2e2e3e;
    --f1-white:  #FFFFFF;
    --f1-grey:   #aaaaaa;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background-color: var(--f1-dark) !important; }

[data-testid="stSidebar"] {
    background-color: var(--f1-panel) !important;
    border-right: 3px solid var(--f1-red);
}

html, body, [class*="css"], p, div, span, label {
    font-family: 'Barlow', sans-serif !important;
    color: var(--f1-white) !important;
}

h1, h2, h3, h4 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    color: var(--f1-white) !important;
}
h1::after {
    content: '';
    display: block;
    width: 60px; height: 4px;
    background: var(--f1-red);
    margin-top: 6px; border-radius: 2px;
}

[data-testid="stMetric"] {
    background-color: var(--f1-panel) !important;
    border: 1px solid var(--f1-border) !important;
    border-left: 4px solid var(--f1-red) !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { color: var(--f1-grey) !important; font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
[data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif !important; font-size: 1.6rem !important; font-weight: 700 !important; }

[data-testid="stFormSubmitButton"] > button {
    background-color: var(--f1-red) !important;
    color: var(--f1-white) !important;
    border: none !important; border-radius: 2px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    text-transform: uppercase !important; letter-spacing: 0.1em !important;
    padding: 10px 24px !important;
}
[data-testid="stFormSubmitButton"] > button:hover { background-color: #c0001e !important; }

input, select, textarea {
    background-color: #0d0d17 !important;
    border: 1px solid var(--f1-border) !important;
    color: var(--f1-white) !important; border-radius: 2px !important;
}

hr { border-color: var(--f1-border) !important; }

/* car image container */
.car-row {
    display: flex; gap: 16px; justify-content: space-between;
    margin: 24px 0;
}
.car-card {
    background: var(--f1-panel);
    border: 1px solid var(--f1-border);
    border-top: 3px solid var(--f1-red);
    border-radius: 4px;
    padding: 16px;
    text-align: center;
    flex: 1;
}
.car-card .team-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800; font-size: 1rem;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--f1-grey); margin-top: 10px;
}
.car-placeholder {
    width: 100%; height: 80px;
    background: linear-gradient(135deg, #1e1e2e 0%, #0d0d17 100%);
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.5rem;
}

/* F1 card */
.f1-card {
    background: var(--f1-panel);
    border: 1px solid var(--f1-border);
    border-top: 3px solid var(--f1-red);
    border-radius: 4px;
    padding: 20px 24px; margin-bottom: 16px;
}
.f1-big-number {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.5rem; font-weight: 800;
    color: var(--f1-red); line-height: 1;
}
.f1-label {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--f1-grey);
}
.f1-pill {
    display: inline-block; background: var(--f1-red); color: white;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 4px 10px; border-radius: 2px;
}
.f1-pill-green { background: #1e7a34; }

/* hero banner */
.hero-banner {
    background: linear-gradient(90deg, #15151E 0%, #1e0008 50%, #15151E 100%);
    border: 1px solid var(--f1-border);
    border-left: 5px solid var(--f1-red);
    border-radius: 4px; padding: 28px 32px;
    margin-bottom: 24px; position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute;
    top: 0; right: 0; bottom: 0; width: 40%;
    background: linear-gradient(90deg, transparent, rgba(232,0,45,0.08));
}

/* dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--f1-border) !important; border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Top nav bar ─────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:#E8002D;padding:10px 24px;margin:-1rem -1rem 1.5rem -1rem;
            display:flex;align-items:center;gap:16px;">
    <span style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                 font-size:1.5rem;color:white;text-transform:uppercase;letter-spacing:0.1em;">
        🏎️ &nbsp; F1 RACE PREDICTOR
    </span>
    <span style="font-size:0.8rem;color:rgba(255,255,255,0.8);margin-left:auto;
                 text-transform:uppercase;letter-spacing:0.08em;">
        2026 Season · Deep Learning Intelligence
    </span>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:8px 0 16px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                    font-size:1.8rem;color:#E8002D;letter-spacing:0.08em;">FORMULA 1</div>
        <div style="font-size:0.7rem;color:#aaa;text-transform:uppercase;letter-spacing:0.15em;">
            Race Intelligence System
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar car image
    car_hero = asset("f1_hero.png") or asset("car_red_bull.png")
    if car_hero:
        st.image(str(car_hero), use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#1e0008;border-radius:4px;padding:20px;text-align:center;
                    font-size:3rem;margin-bottom:8px;">🏎️</div>
        """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "nav",
        ["🏠  Overview", "⏱  Lap Time", "🔧  Pit Stop", "🏁  Final Position"],
        label_visibility="collapsed",
    )

    st.divider()
    auto_refresh = st.toggle("🔄  Live auto-refresh (30s)", value=False)

    st.markdown("""
    <div style="margin-top:24px;padding:12px;background:#0d0d17;border-radius:4px;
                border-left:3px solid #E8002D;">
        <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#aaa;">Stack</div>
        <div style="font-size:0.8rem;color:#fff;margin-top:6px;line-height:1.9;">
            PyTorch · FastAPI<br>MLflow · FastF1<br>Streamlit · Docker
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── API Status ───────────────────────────────────────────────────────────────

health = get_health()
render_api_status(health)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Overview":

    # Hero banner
    hero = asset("f1_hero.png")
    if hero:
        st.image(str(hero), use_container_width=True)
    else:
        st.markdown("""
        <div class="hero-banner">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:3rem;
                        font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">
                Race Intelligence<br>
                <span style="color:#E8002D;">Powered by Deep Learning</span>
            </div>
            <div style="color:#aaa;margin-top:8px;font-size:1rem;">
                LSTM · FCNN · MLP &nbsp;|&nbsp; 75 Seasons of F1 Data (1950–2024)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.title("Prediction Models")

    # Model cards with car images
    c1, c2, c3 = st.columns(3)

    with c1:
        car_img = asset("car_red_bull.png")
        if car_img:
            st.image(str(car_img), use_container_width=True)
        else:
            st.markdown('<div class="car-placeholder">🏎️</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="f1-card">
            <div class="f1-label">Model 01</div>
            <h2 style="color:#fff!important;font-size:1.5rem!important;margin:4px 0 8px;">
                LAP TIME LSTM
            </h2>
            <div style="color:#aaa;font-size:0.85rem;line-height:1.6;">
                Two-layer LSTM with driver &amp; circuit embeddings.<br>
                Predicts next lap time in seconds from last 5 laps.
            </div>
            <div style="margin-top:14px;"><span class="f1-pill">RMSE ~6 s</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        car_img = asset("car_ferrari.png")
        if car_img:
            st.image(str(car_img), use_container_width=True)
        else:
            st.markdown('<div class="car-placeholder">🏎️</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="f1-card">
            <div class="f1-label">Model 02</div>
            <h2 style="color:#fff!important;font-size:1.5rem!important;margin:4px 0 8px;">
                PIT STOP FCNN
            </h2>
            <div style="color:#aaa;font-size:0.85rem;line-height:1.6;">
                Fully-connected binary classifier.<br>
                Predicts if a driver will pit on the next lap.
            </div>
            <div style="margin-top:14px;"><span class="f1-pill">F1 0.0955 · AUC 0.56</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        car_img = asset("car_mercedes.png")
        if car_img:
            st.image(str(car_img), use_container_width=True)
        else:
            st.markdown('<div class="car-placeholder">🏎️</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="f1-card">
            <div class="f1-label">Model 03</div>
            <h2 style="color:#fff!important;font-size:1.5rem!important;margin:4px 0 8px;">
                POSITION MLP
            </h2>
            <div style="color:#aaa;font-size:0.85rem;line-height:1.6;">
                Multi-layer perceptron regression.<br>
                Estimates final race finishing position.
            </div>
            <div style="margin-top:14px;"><span class="f1-pill">MAE ~3.17 positions</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Stats row
    st.subheader("Training Data")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Seasons", "75",    "1950 – 2024")
    m2.metric("Lap sequences", "533K+", "LSTM training")
    m3.metric("Pit events",    "10.7K+","labelled")
    m4.metric("Race results",  "4.2K+", "cleaned")

    st.divider()
    st.info("▶  Start the API first:  `uvicorn src.api.main:app --reload --port 8000`")


# ─────────────────────────────────────────────────────────────────────────────
# LAP TIME
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⏱  Lap Time":
    # Car banner
    car_img = asset("car_red_bull.png") or asset("f1_hero.png")
    if car_img:
        st.image(str(car_img), use_container_width=True)

    st.title("Lap Time Predictor")
    st.markdown("<div style='color:#aaa;margin-top:-12px;margin-bottom:24px;'>Enter the last 5 laps to predict the next lap time.</div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='f1-card'>", unsafe_allow_html=True)
        st.markdown("### Last 5 Laps")
        with st.form("lap_time_form"):
            laps_input = []
            for i in range(1, 6):
                c1, c2 = st.columns(2)
                lt = c1.number_input(f"Lap {i} — time (s)", 40.0, 200.0, 90.0 + i * 0.3, key=f"lt_{i}")
                ln = c2.number_input(f"Lap {i} — number",  1,    80,    i,               key=f"ln_{i}")
                laps_input.append({"lap_time_sec": lt, "lap": ln})
            st.markdown("---")
            st.markdown("### Driver & Circuit")
            driver_id  = st.number_input("Driver ID (encoded)", min_value=0, value=0)
            circuit_id = st.number_input("Circuit ID (encoded)", min_value=0, value=0)
            submitted  = st.form_submit_button("PREDICT NEXT LAP TIME", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if submitted:
            with st.spinner("Running inference …"):
                result = predict_lap_time(laps_input, int(driver_id), int(circuit_id))
            if result and "error" not in result:
                pred = result["predicted_lap_time_sec"]
                st.plotly_chart(lap_time_gauge(pred), use_container_width=True)
                st.markdown(f"""
                <div class="f1-card" style="text-align:center;">
                    <div class="f1-label">Predicted Next Lap</div>
                    <div class="f1-big-number">{pred:.3f} s</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(result.get("error", "Prediction failed — is the API running?"))
        else:
            st.markdown("""
            <div class="f1-card" style="text-align:center;padding:48px;">
                <div style="font-size:3rem;">⏱</div>
                <div class="f1-label" style="margin-top:12px;">Enter lap data and click predict</div>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Lap History")
        history = [{"lap": i, "lap_time_sec": 90.0 + (i % 4) * 0.5 - i * 0.05} for i in range(1, 16)]
        st.plotly_chart(lap_history_chart(history), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PIT STOP
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔧  Pit Stop":
    car_img = asset("car_ferrari.png") or asset("f1_hero.png")
    if car_img:
        st.image(str(car_img), use_container_width=True)

    st.title("Pit Stop Predictor")
    st.markdown("<div style='color:#aaa;margin-top:-12px;margin-bottom:24px;'>Will the driver pit on the <b>next lap</b>?</div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='f1-card'>", unsafe_allow_html=True)
        st.markdown("### Race State")
        with st.form("pit_stop_form"):
            lap             = st.number_input("Current lap",            1,  80, 20)
            cumulative_pits = st.number_input("Cumulative pit stops",   0,  10,  1)
            prev_lap_ms     = st.number_input("Previous lap (ms)", min_value=1.0, value=90500.0, step=100.0)
            curr_lap_ms     = st.number_input("Current lap (ms)",  min_value=1.0, value=91800.0, step=100.0)
            st.slider("Estimated tyre age (laps)", 1, 40, 15, help="Informational — not sent to model yet")
            submitted = st.form_submit_button("PREDICT PIT STOP", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if submitted:
            with st.spinner("Running inference …"):
                result = predict_pit_stop(int(lap), int(cumulative_pits), prev_lap_ms, curr_lap_ms)
            if result and "error" not in result:
                prob      = result["probability"]
                will_pit  = result["will_pit"]
                threshold = result["threshold_used"]
                st.plotly_chart(pit_probability_bar(prob, threshold), use_container_width=True)
                if will_pit:
                    st.markdown(f"""
                    <div class="f1-card" style="text-align:center;border-top-color:#E8002D;">
                        <div class="f1-pill">🔧 PIT STOP PREDICTED</div>
                        <div class="f1-big-number" style="margin-top:12px;">{prob*100:.1f}%</div>
                        <div class="f1-label">Pit probability</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="f1-card" style="text-align:center;border-top-color:#1e7a34;">
                        <div class="f1-pill f1-pill-green">✅ STAY OUT</div>
                        <div class="f1-big-number" style="color:#1e7a34;margin-top:12px;">{prob*100:.1f}%</div>
                        <div class="f1-label">Pit probability</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(result.get("error", "Prediction failed — is the API running?"))
        else:
            st.markdown("""
            <div class="f1-card" style="text-align:center;padding:48px;">
                <div style="font-size:3rem;">🔧</div>
                <div class="f1-label" style="margin-top:12px;">Enter race state and click predict</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FINAL POSITION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏁  Final Position":
    car_img = asset("car_mercedes.png") or asset("f1_hero.png")
    if car_img:
        st.image(str(car_img), use_container_width=True)

    st.title("Final Position Predictor")
    st.markdown("<div style='color:#aaa;margin-top:-12px;margin-bottom:24px;'>Estimate a driver's race finishing position.</div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='f1-card'>", unsafe_allow_html=True)
        st.markdown("### Race Result Data")
        with st.form("position_form"):
            grid            = st.number_input("Grid / qualifying position", 1, 20,  5)
            laps            = st.number_input("Laps completed",             1, 80, 57)
            race_time_ms    = st.number_input("Total race time (ms)", min_value=1.0,
                                               value=5_400_000.0, step=10_000.0, format="%.0f")
            fastest_lap_sec = st.number_input("Fastest lap (s)", 40.0, 200.0, 88.5)
            submitted       = st.form_submit_button("PREDICT FINISH POSITION", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if submitted:
            with st.spinner("Running inference …"):
                result = predict_position(int(grid), int(laps), race_time_ms, fastest_lap_sec)
            if result and "error" not in result:
                pos     = result["predicted_position_rounded"]
                pos_raw = result["predicted_position"]
                podium_colors = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
                color = podium_colors.get(pos, "#E8002D")
                st.plotly_chart(position_podium(pos), use_container_width=True)
                st.markdown(f"""
                <div class="f1-card" style="text-align:center;">
                    <div class="f1-label">Predicted Finish</div>
                    <div class="f1-big-number" style="color:{color};font-size:5rem;">P{pos}</div>
                    <div class="f1-label">Raw score: {pos_raw:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(result.get("error", "Prediction failed — is the API running?"))
        else:
            st.markdown("""
            <div class="f1-card" style="text-align:center;padding:48px;">
                <div style="font-size:3rem;">🏁</div>
                <div class="f1-label" style="margin-top:12px;">Enter race data and click predict</div>
            </div>
            """, unsafe_allow_html=True)

        # Leaderboard
        st.subheader("Sample Leaderboard")
        st.dataframe(
            [
                {"POS": "P1", "DRIVER": "Verstappen", "TEAM": "Red Bull",  "GAP": "LEADER"},
                {"POS": "P2", "DRIVER": "Leclerc",    "TEAM": "Ferrari",   "GAP": "+4.2s"},
                {"POS": "P3", "DRIVER": "Hamilton",   "TEAM": "Mercedes",  "GAP": "+9.1s"},
                {"POS": "P4", "DRIVER": "Norris",     "TEAM": "McLaren",   "GAP": "+12.7s"},
                {"POS": "P5", "DRIVER": "Russell",    "TEAM": "Mercedes",  "GAP": "+18.3s"},
            ],
            use_container_width=True, hide_index=True,
        )

# ─── Auto-refresh ─────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(30)
    st.rerun()
