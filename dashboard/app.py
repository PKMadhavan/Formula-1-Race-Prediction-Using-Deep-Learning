"""
F1 Race Predictor — Streamlit Dashboard
Styled to match the official Formula 1 website aesthetic.

Run:
    python3.12 -m streamlit run dashboard/app.py

2026 Grid: 22 drivers, 11 teams
    Teams: Red Bull, Ferrari, Mercedes, McLaren, Aston Martin,
           Alpine, Williams, Haas, Racing Bulls, Audi (ex-Sauber), Cadillac F1

Car images (optional — drop PNG files in dashboard/assets/):
    f1_hero.png            wide hero banner
    car_red_bull.png       Red Bull Racing
    car_ferrari.png        Ferrari
    car_mercedes.png       Mercedes
    car_mclaren.png        McLaren
    car_aston.png          Aston Martin
    car_alpine.png         Alpine
    car_williams.png       Williams
    car_haas.png           Haas
    car_rb.png             Racing Bulls
    car_audi.png           Audi (formerly Sauber)
    car_cadillac.png       Cadillac F1 (new 11th team)
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from dashboard.api_client import (
    get_health, predict_lap_time, predict_pit_stop, predict_position,
)
from dashboard.components import (
    lap_history_chart, lap_time_gauge, pit_probability_bar,
    position_podium, render_api_status,
)
from dashboard.live_2026 import (
    DRIVERS_2026, TEAM_COLORS,
    get_2026_schedule, get_next_race, get_last_race,
    load_qualifying_results, predict_race_outcome, predict_next_lap_times,
)

ASSETS = Path(__file__).parent / "assets"

TEAM_CAR_ASSET = {
    "Red Bull Racing": "car_red_bull.png",
    "Ferrari":         "car_ferrari.png",
    "Mercedes":        "car_mercedes.png",
    "McLaren":         "car_mclaren.png",
    "Aston Martin":    "car_aston.png",
    "Alpine":          "car_alpine.png",
    "Williams":        "car_williams.png",
    "Haas":            "car_haas.png",
    "Racing Bulls":    "car_rb.png",
    "Audi":            "car_audi.png",       # formerly Sauber
    "Cadillac F1":     "car_cadillac.png",   # new 11th team 2026
}


def asset(name: str) -> Path | None:
    p = ASSETS / name
    return p if p.exists() else None


def team_car_image(team: str) -> Path | None:
    filename = TEAM_CAR_ASSET.get(team)
    return asset(filename) if filename else None


def team_color_bar(team: str, height: int = 4) -> str:
    color = TEAM_COLORS.get(team, "#888")
    return f'<div style="background:{color};height:{height}px;border-radius:2px;margin-bottom:8px;"></div>'


# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Predictor 2026",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@400;500;600&display=swap');
:root {
    --f1-red:#E8002D; --f1-dark:#15151E; --f1-panel:#1e1e2e;
    --f1-border:#2e2e3e; --f1-white:#FFFFFF; --f1-grey:#aaaaaa;
}
[data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background-color:var(--f1-dark)!important;}
[data-testid="stSidebar"]{background-color:var(--f1-panel)!important;border-right:3px solid var(--f1-red);}
html,body,[class*="css"],p,div,span,label{font-family:'Barlow',sans-serif!important;color:var(--f1-white)!important;}
h1,h2,h3,h4{font-family:'Barlow Condensed',sans-serif!important;font-weight:800!important;text-transform:uppercase!important;letter-spacing:0.04em!important;color:var(--f1-white)!important;}
h1::after{content:'';display:block;width:60px;height:4px;background:var(--f1-red);margin-top:6px;border-radius:2px;}
[data-testid="stMetric"]{background-color:var(--f1-panel)!important;border:1px solid var(--f1-border)!important;border-left:4px solid var(--f1-red)!important;border-radius:4px!important;padding:12px 16px!important;}
[data-testid="stMetricLabel"]{color:var(--f1-grey)!important;font-size:0.75rem!important;text-transform:uppercase!important;letter-spacing:0.08em!important;}
[data-testid="stMetricValue"]{font-family:'Barlow Condensed',sans-serif!important;font-size:1.6rem!important;font-weight:700!important;}
[data-testid="stFormSubmitButton"]>button{background-color:var(--f1-red)!important;color:var(--f1-white)!important;border:none!important;border-radius:2px!important;font-family:'Barlow Condensed',sans-serif!important;font-weight:700!important;font-size:1rem!important;text-transform:uppercase!important;letter-spacing:0.1em!important;padding:10px 24px!important;}
[data-testid="stFormSubmitButton"]>button:hover{background-color:#c0001e!important;}
input,select,textarea{background-color:#0d0d17!important;border:1px solid var(--f1-border)!important;color:var(--f1-white)!important;border-radius:2px!important;}
hr{border-color:var(--f1-border)!important;}
.f1-card{background:var(--f1-panel);border:1px solid var(--f1-border);border-top:3px solid var(--f1-red);border-radius:4px;padding:20px 24px;margin-bottom:16px;}
.f1-big-number{font-family:'Barlow Condensed',sans-serif;font-size:3.5rem;font-weight:800;color:var(--f1-red);line-height:1;}
.f1-label{font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--f1-grey);}
.f1-pill{display:inline-block;background:var(--f1-red);color:white;font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.1em;padding:4px 10px;border-radius:2px;}
.f1-pill-green{background:#1e7a34;}
.f1-pill-gold{background:#c9a227;}
.driver-card{background:var(--f1-panel);border:1px solid var(--f1-border);border-radius:4px;padding:12px 16px;margin-bottom:8px;display:flex;align-items:center;gap:12px;}
.driver-pos{font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:1.6rem;min-width:36px;text-align:center;}
.driver-name{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:1rem;text-transform:uppercase;}
.driver-team{font-size:0.75rem;color:var(--f1-grey);}
.car-placeholder{width:100%;height:72px;display:flex;align-items:center;justify-content:center;font-size:2.5rem;background:linear-gradient(135deg,#1e1e2e,#0d0d17);border-radius:4px;}
[data-testid="stDataFrame"]{border:1px solid var(--f1-border)!important;border-radius:4px!important;}
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
    <span style="background:white;color:#E8002D;font-family:'Barlow Condensed',sans-serif;
                 font-weight:800;font-size:0.8rem;padding:2px 8px;border-radius:2px;
                 text-transform:uppercase;letter-spacing:0.08em;">
        2026 SEASON
    </span>
    <span style="font-size:0.75rem;color:rgba(255,255,255,0.8);margin-left:auto;
                 text-transform:uppercase;letter-spacing:0.08em;">
        Live Deep Learning Predictions
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
            Race Intelligence · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    hero = asset("f1_hero.png")
    if hero:
        st.image(str(hero), width='stretch')
    else:
        st.markdown('<div class="car-placeholder" style="height:80px;margin-bottom:8px;">🏎️</div>',
                    unsafe_allow_html=True)

    st.divider()
    page = st.radio("nav", [
        "🏠  Overview",
        "🏆  2026 Live Race",
        "⏱  Lap Time",
        "🔧  Pit Stop",
        "🏁  Final Position",
    ], label_visibility="collapsed")

    st.divider()
    auto_refresh = st.toggle("🔄  Live auto-refresh (30s)", value=False)
    st.markdown("""
    <div style="margin-top:24px;padding:12px;background:#0d0d17;border-radius:4px;border-left:3px solid #E8002D;">
        <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#aaa;">Stack</div>
        <div style="font-size:0.8rem;color:#fff;margin-top:6px;line-height:1.9;">
            PyTorch · FastAPI<br>FastF1 · MLflow<br>Streamlit · Docker
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── API Status ───────────────────────────────────────────────────────────────
health = get_health()
render_api_status(health)
st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    hero = asset("f1_hero.png")
    if hero:
        st.image(str(hero), width='stretch')
    else:
        st.markdown("""
        <div style="background:linear-gradient(90deg,#15151E,#1e0008,#15151E);
                    border:1px solid #2e2e3e;border-left:5px solid #E8002D;
                    border-radius:4px;padding:28px 32px;margin-bottom:24px;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:3rem;
                        font-weight:800;text-transform:uppercase;">
                Race Intelligence<br>
                <span style="color:#E8002D;">2026 Season · Live Predictions</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.title("Prediction Models")
    c1, c2, c3 = st.columns(3)
    for col, (team, label, desc, badge) in zip([c1, c2, c3], [
        ("Red Bull Racing", "LAP TIME LSTM",  "Predicts next lap time from last 5 laps using LSTM + embeddings.", "RMSE ~6 s"),
        ("Ferrari",         "PIT STOP FCNN",  "Binary classifier — will the driver pit on the next lap?",        "F1 0.0955"),
        ("Mercedes",        "POSITION MLP",   "Regression model estimating final race finishing position.",       "MAE ~3.2"),
    ]):
        with col:
            img = team_car_image(team)
            if img:
                st.image(str(img), width='stretch')
            else:
                color = TEAM_COLORS[team]
                st.markdown(f'<div class="car-placeholder" style="border-top:3px solid {color};">🏎️</div>',
                            unsafe_allow_html=True)
            st.markdown(team_color_bar(team, 3), unsafe_allow_html=True)
            st.markdown(f"""
            <div class="f1-card" style="margin-top:0;">
                <div class="f1-label">{team}</div>
                <h3 style="color:#fff!important;font-size:1.3rem!important;margin:4px 0 8px;">{label}</h3>
                <div style="color:#aaa;font-size:0.85rem;line-height:1.6;">{desc}</div>
                <div style="margin-top:14px;"><span class="f1-pill">{badge}</span></div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.subheader(f"2026 Grid — All {len(DRIVERS_2026)} Drivers · {len(TEAM_COLORS)} Teams")

    cols = st.columns(4)
    for i, driver in enumerate(DRIVERS_2026):
        with cols[i % 4]:
            color    = TEAM_COLORS.get(driver["team"], "#888")
            new_badge = '<span style="background:#E8002D;color:white;font-size:0.6rem;padding:1px 5px;border-radius:2px;margin-left:4px;font-weight:700;">NEW</span>' if driver.get("new") else ""
            st.markdown(f"""
            <div style="background:#1e1e2e;border:1px solid #2e2e3e;border-left:3px solid {color};
                        border-radius:4px;padding:10px 12px;margin-bottom:8px;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                            font-size:0.95rem;text-transform:uppercase;">
                    {driver['name']}{new_badge}
                </div>
                <div style="font-size:0.72rem;color:#aaa;">{driver['team']} &nbsp;·&nbsp; #{driver['number']}</div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: 2026 Live Race
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏆  2026 Live Race":
    st.title("2026 Live Race Prediction")

    from dashboard.live_2026 import MODEL_DISCLAIMER
    st.warning(MODEL_DISCLAIMER)

    with st.spinner("Fetching 2026 F1 schedule …"):
        schedule = get_2026_schedule()
    next_race = get_next_race(schedule)

    if next_race:
        event_date = str(next_race.get("EventDate", ""))[:10]
        st.markdown(f"""
        <div style="background:linear-gradient(90deg,#1e0008,#15151E);
                    border:1px solid #2e2e3e;border-left:5px solid #E8002D;
                    border-radius:4px;padding:20px 28px;margin-bottom:24px;">
            <div class="f1-label">Next Race · Round {int(next_race.get('RoundNumber',0))}</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                        font-size:2.2rem;text-transform:uppercase;margin:4px 0;">
                {next_race.get('EventName','—')}
            </div>
            <div style="color:#aaa;font-size:0.95rem;">
                📍 {next_race.get('Location','—')}, {next_race.get('Country','—')}
                &nbsp;&nbsp;|&nbsp;&nbsp;
                📅 {event_date}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Season calendar strip
    st.subheader("2026 Season Calendar")
    cal_cols = st.columns(6)
    for i, (_, row) in enumerate(schedule.iterrows()):
        date_str = str(row.get("EventDate",""))[:10]
        is_next  = next_race and row.get("RoundNumber") == next_race.get("RoundNumber")
        bg       = "#E8002D" if is_next else "#1e1e2e"
        border   = "2px solid #E8002D" if is_next else "1px solid #2e2e3e"
        with cal_cols[i % 6]:
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:4px;
                        padding:8px 10px;margin-bottom:8px;text-align:center;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                            font-size:0.8rem;text-transform:uppercase;">
                    R{int(row.get('RoundNumber',0))}
                </div>
                <div style="font-size:0.65rem;color:{'white' if is_next else '#aaa'};">
                    {row.get('Country','?')}
                </div>
                <div style="font-size:0.6rem;color:{'white' if is_next else '#555'};">
                    {date_str}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Predicted race outcome
    col_pred, col_lap = st.columns([1, 1], gap="large")

    with col_pred:
        st.subheader("Predicted Race Outcome")
        st.caption("Based on 2026 grid · PositionMLP model")

        # Try qualifying — fall back to assumed grid order
        round_num = int(next_race.get("RoundNumber", 1)) if next_race else 1
        with st.spinner("Loading qualifying data …"):
            quali = load_qualifying_results(2026, round_num)

        if quali is not None:
            grid_order = [{"code": row.get("Abbreviation",""), "name": row.get("FullName",""),
                           "team": row.get("TeamName",""), "grid": int(row.get("grid",i+1))}
                          for i, row in quali.iterrows()]
        else:
            grid_order = [{"code": d["code"], "name": d["name"],
                           "team": d["team"], "grid": i+1}
                          for i, d in enumerate(DRIVERS_2026)]

        # Load model store for predictions
        try:
            from src.api.model_loader import store
            if not any(store.models_status.values()):
                store.load_all()
        except Exception:
            store = None

        if store:
            predicted = predict_race_outcome(grid_order, store)
        else:
            predicted = [{**d, "predicted_position": d["grid"] + 0.5, "predicted_finish": d["grid"]}
                         for d in grid_order]

        # Podium
        podium_colors = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
        p_cols = st.columns(3)
        for col, pos in zip(p_cols, [1, 2, 3]):
            driver = next((d for d in predicted if d["predicted_finish"] == pos), None)
            if driver:
                color = podium_colors[pos]
                team_color = TEAM_COLORS.get(driver["team"], "#888")
                with col:
                    img = team_car_image(driver["team"])
                    if img:
                        st.image(str(img), width='stretch')
                    else:
                        st.markdown(
                            f'<div class="car-placeholder" style="height:56px;border-top:3px solid {team_color};">🏎️</div>',
                            unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="text-align:center;padding:8px;background:#1e1e2e;
                                border-radius:4px;border-top:3px solid {color};">
                        <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                                    font-size:1.6rem;color:{color};">P{pos}</div>
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.9rem;
                                    font-weight:700;text-transform:uppercase;">{driver['code']}</div>
                        <div style="font-size:0.7rem;color:#aaa;">{driver['team']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Full race prediction table
        st.markdown("**Full Race Prediction**")
        for driver in predicted:
            pos       = driver["predicted_finish"]
            color     = podium_colors.get(pos, TEAM_COLORS.get(driver["team"], "#888"))
            tc        = TEAM_COLORS.get(driver["team"], "#888")
            new_badge = '<span style="background:#555;color:white;font-size:0.6rem;padding:1px 5px;border-radius:2px;margin-left:4px;">NEW</span>' if driver.get("new_driver") else ""
            st.markdown(f"""
            <div class="driver-card" style="border-left:4px solid {tc};">
                <div class="driver-pos" style="color:{color};">P{pos}</div>
                <div>
                    <div class="driver-name">{driver['code']} · {driver['name']}{new_badge}</div>
                    <div class="driver-team">{driver['team']}</div>
                </div>
                <div style="margin-left:auto;font-family:'Barlow Condensed',sans-serif;
                            font-size:0.85rem;color:#aaa;">
                    score {driver['predicted_position']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_lap:
        st.subheader("Predicted Lap Times")
        st.caption("Top 10 drivers · LapTimeLSTM model")

        if store:
            lap_preds = predict_next_lap_times(store, num_drivers=10)
        else:
            lap_preds = [{"driver": d["code"], "name": d["name"], "team": d["team"],
                          "predicted_lap_time_sec": 88.5 + i * 0.3}
                         for i, d in enumerate(DRIVERS_2026[:10])]

        fastest = lap_preds[0]["predicted_lap_time_sec"] if lap_preds else 88.5
        for rank, lp in enumerate(lap_preds):
            tc  = TEAM_COLORS.get(lp["team"], "#888")
            gap = lp["predicted_lap_time_sec"] - fastest
            gap_str = "FASTEST" if rank == 0 else f"+{gap:.3f}s"
            pill_cls = "f1-pill-gold" if rank == 0 else "f1-pill"
            st.markdown(f"""
            <div style="background:#1e1e2e;border:1px solid #2e2e3e;border-left:3px solid {tc};
                        border-radius:4px;padding:10px 14px;margin-bottom:6px;
                        display:flex;align-items:center;gap:12px;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                            font-size:1rem;min-width:32px;color:#aaa;">P{rank+1}</div>
                <div style="flex:1;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                                text-transform:uppercase;">{lp['driver']} · {lp['name']}</div>
                    <div style="font-size:0.72rem;color:#aaa;">{lp['team']}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;
                                font-size:1.1rem;">{lp['predicted_lap_time_sec']:.3f}s</div>
                    <span class="{pill_cls}" style="font-size:0.7rem;">{gap_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Team Car Gallery")
        t_cols = st.columns(5)
        for i, (team, color) in enumerate(TEAM_COLORS.items()):
            with t_cols[i % 5]:
                img = team_car_image(team)
                if img:
                    st.image(str(img), width='stretch')
                else:
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,{color}22,#0d0d17);
                                border:1px solid {color};border-radius:4px;
                                height:64px;display:flex;align-items:center;
                                justify-content:center;font-size:1.8rem;
                                margin-bottom:4px;">🏎️</div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="text-align:center;font-family:'Barlow Condensed',sans-serif;
                            font-size:0.72rem;font-weight:700;text-transform:uppercase;
                            color:{color};margin-bottom:8px;">{team}</div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Lap Time
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⏱  Lap Time":
    img = team_car_image("Red Bull Racing") or asset("f1_hero.png")
    if img: st.image(str(img), width='stretch')
    st.markdown(team_color_bar("Red Bull Racing"), unsafe_allow_html=True)

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
                ln = c2.number_input(f"Lap {i} — number",  1, 80, i, key=f"ln_{i}")
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
                st.markdown(f'<div class="f1-card" style="text-align:center;"><div class="f1-label">Predicted Next Lap</div><div class="f1-big-number">{pred:.3f} s</div></div>', unsafe_allow_html=True)
            else:
                st.error(result.get("error","Prediction failed — is the API running?"))
        else:
            st.markdown('<div class="f1-card" style="text-align:center;padding:48px;"><div style="font-size:3rem;">⏱</div><div class="f1-label" style="margin-top:12px;">Enter lap data and click predict</div></div>', unsafe_allow_html=True)
        st.subheader("Lap History")
        history = [{"lap": i, "lap_time_sec": 90.0 + (i % 4)*0.5 - i*0.05} for i in range(1, 16)]
        st.plotly_chart(lap_history_chart(history), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Pit Stop
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔧  Pit Stop":
    img = team_car_image("Ferrari") or asset("f1_hero.png")
    if img: st.image(str(img), width='stretch')
    st.markdown(team_color_bar("Ferrari"), unsafe_allow_html=True)

    st.title("Pit Stop Predictor")
    st.markdown("<div style='color:#aaa;margin-top:-12px;margin-bottom:24px;'>Will the driver pit on the <b>next lap</b>?</div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")
    with col_form:
        st.markdown("<div class='f1-card'>", unsafe_allow_html=True)
        st.markdown("### Race State")
        with st.form("pit_stop_form"):
            lap             = st.number_input("Current lap",          1, 80, 20)
            cumulative_pits = st.number_input("Cumulative pit stops", 0, 10,  1)
            prev_lap_ms     = st.number_input("Previous lap (ms)", min_value=1.0, value=90500.0, step=100.0)
            curr_lap_ms     = st.number_input("Current lap (ms)",  min_value=1.0, value=91800.0, step=100.0)
            st.slider("Tyre age (laps)", 1, 40, 15, help="Informational")
            submitted = st.form_submit_button("PREDICT PIT STOP", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if submitted:
            with st.spinner("Running inference …"):
                result = predict_pit_stop(int(lap), int(cumulative_pits), prev_lap_ms, curr_lap_ms)
            if result and "error" not in result:
                prob, will_pit, threshold = result["probability"], result["will_pit"], result["threshold_used"]
                st.plotly_chart(pit_probability_bar(prob, threshold), use_container_width=True)
                css_class = "f1-pill" if will_pit else "f1-pill-green"
                label     = "🔧 PIT STOP PREDICTED" if will_pit else "✅ STAY OUT"
                color     = "#E8002D" if will_pit else "#1e7a34"
                st.markdown(f'<div class="f1-card" style="text-align:center;border-top-color:{color};"><span class="{css_class}">{label}</span><div class="f1-big-number" style="color:{color};margin-top:12px;">{prob*100:.1f}%</div><div class="f1-label">Pit probability</div></div>', unsafe_allow_html=True)
            else:
                st.error(result.get("error","Prediction failed — is the API running?"))
        else:
            st.markdown('<div class="f1-card" style="text-align:center;padding:48px;"><div style="font-size:3rem;">🔧</div><div class="f1-label" style="margin-top:12px;">Enter race state and click predict</div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Final Position
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏁  Final Position":
    img = team_car_image("Mercedes") or asset("f1_hero.png")
    if img: st.image(str(img), width='stretch')
    st.markdown(team_color_bar("Mercedes"), unsafe_allow_html=True)

    st.title("Final Position Predictor")
    st.markdown("<div style='color:#aaa;margin-top:-12px;margin-bottom:24px;'>Estimate a driver's race finishing position.</div>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")
    with col_form:
        st.markdown("<div class='f1-card'>", unsafe_allow_html=True)
        st.markdown("### Race Result Data")

        driver_select = st.selectbox("Driver", [f"{d['name']} ({d['team']})" for d in DRIVERS_2026])
        selected_driver = DRIVERS_2026[[f"{d['name']} ({d['team']})" for d in DRIVERS_2026].index(driver_select)]
        team_color = TEAM_COLORS.get(selected_driver["team"], "#888")
        st.markdown(f'<div style="background:{team_color}22;border-left:3px solid {team_color};padding:8px 12px;border-radius:2px;margin-bottom:12px;font-size:0.85rem;">#{selected_driver["number"]} · {selected_driver["team"]}</div>', unsafe_allow_html=True)

        # Show team car if available
        img2 = team_car_image(selected_driver["team"])
        if img2:
            st.image(str(img2), width='stretch')

        with st.form("position_form"):
            grid            = st.number_input("Grid / qualifying position", 1, 20, 5)
            laps            = st.number_input("Laps completed", 1, 80, 57)
            race_time_ms    = st.number_input("Total race time (ms)", min_value=1.0, value=5_400_000.0, step=10_000.0, format="%.0f")
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
                color = podium_colors.get(pos, team_color)
                st.plotly_chart(position_podium(pos), use_container_width=True)
                st.markdown(f'<div class="f1-card" style="text-align:center;"><div class="f1-label">Predicted Finish</div><div class="f1-big-number" style="color:{color};font-size:5rem;">P{pos}</div><div class="f1-label">Raw score: {pos_raw:.2f}</div></div>', unsafe_allow_html=True)
            else:
                st.error(result.get("error","Prediction failed — is the API running?"))
        else:
            st.markdown('<div class="f1-card" style="text-align:center;padding:48px;"><div style="font-size:3rem;">🏁</div><div class="f1-label" style="margin-top:12px;">Select a driver and click predict</div></div>', unsafe_allow_html=True)

        st.subheader("Sample Leaderboard")
        st.dataframe([
            {"POS":"P1","DRIVER":"Verstappen","TEAM":"Red Bull","GAP":"LEADER"},
            {"POS":"P2","DRIVER":"Norris",    "TEAM":"McLaren", "GAP":"+4.2s"},
            {"POS":"P3","DRIVER":"Leclerc",   "TEAM":"Ferrari", "GAP":"+9.1s"},
            {"POS":"P4","DRIVER":"Hamilton",  "TEAM":"Ferrari", "GAP":"+12.7s"},
            {"POS":"P5","DRIVER":"Russell",   "TEAM":"Mercedes","GAP":"+18.3s"},
        ], use_container_width=True, hide_index=True)

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()
