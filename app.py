import sys
import asyncio

# Prevent WinError 10054 on Windows socket reset
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Load trained pipeline
pipeline = joblib.load('models/best_pipeline.pkl')

FAVICON = """data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%2306b6d4"><path d="M12 2a9 9 0 0 0-9 9c0 3.87 2.45 7.17 5.92 8.42.45.08.62-.2.62-.44v-1.54c-2.42.53-2.93-1.17-2.93-1.17-.4-.99-.97-1.26-.97-1.26-.79-.54.06-.53.06-.53.87.06 1.33.9 1.33.9.77 1.33 2.03.95 2.53.72.08-.56.3-.95.55-1.17-1.93-.22-3.96-.97-3.96-4.31 0-.95.34-1.73.9-2.34-.09-.22-.39-1.11.09-2.31 0 0 .73-.23 2.4 1.12a8.38 8.38 0 0 1 4.38 0c1.67-1.35 2.4-1.12 2.4-1.12.48 1.2.18 2.09.09 2.31.56.61.9 1.39.9 2.34 0 3.35-2.03 4.09-3.97 4.31.31.27.59.8.59 1.62v2.4c0 .24.16.53.62.44A9.003 9.003 0 0 0 21 11a9 9 0 0 0-9-9z"/></svg>"""

st.set_page_config(
    page_title="NEXUS // CareerPulse AI",
    page_icon=FAVICON,
    layout="wide"
)

# Navigation State Management
if "current_page" not in st.session_state:
    st.session_state.current_page = "overview"

def navigate_to(page_name):
    st.session_state.current_page = page_name

# Inline SVG Vector Helper (zero external font dependencies)
def svg_icon(path_d, color="#38bdf8", size=16, viewBox="0 0 24 24"):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewBox}" width="{size}" height="{size}" fill="{color}" style="vertical-align: -2px; display: inline-block;"><path d="{path_d}"/></svg>"""

# Vector Path Constants
ICO_CHIP = "M6 2v2H4c-.55 0-1 .45-1 1v2H1v2h2v2H1v2h2v2H1v2h2v2c0 .55.45 1 1 1h2v2h2v-2h2v2h2v-2h2v2h2v-2h2c.55 0 1-.45 1-1v-2h2v-2h-2v-2h2v-2h-2v-2h2V7h-2V5c0-.55-.45-1-1-1h-2V2h-2v2h-2V2h-2v2H8V2H6zm2 4h8v8H8V6zm2 2v4h4V8h-4z"
ICO_SEARCH = "M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"
ICO_ID = "M20 4H4c-1.11 0-1.99.89-1.99 2L2 18c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V6c0-1.11-.89-2-2-2zm-9 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm6 10H5v-.5c0-1.66 3.33-2.5 5-2.5s5 .84 5 2.5v.5zm3-4h-5v-1h5v1zm0-2h-5v-1h5v1zm0-2h-5V9h5v1z"
ICO_LAYERS = "M11.99 18.54l-7.37-5.73L3 14.07l9 7 9-7-1.63-1.27-7.38 5.74zM12 16l7.36-5.73L21 9.07l-9-7-9 7 1.63 1.2L12 16z"
ICO_CHECK = "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
ICO_BAN = "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8 0-1.85.63-3.55 1.69-4.9L16.9 18.31C15.55 19.37 13.85 20 12 20zm6.31-3.1L7.1 5.69C8.45 4.63 10.15 4 12 4c4.42 0 8 3.58 8 8 0 1.85-.63 3.55-1.69 4.9z"
ICO_BRIEFCASE = "M20 6h-4V4c0-1.11-.89-2-2-2h-4c-1.11 0-2 .89-2 2v2H4c-1.11 0-1.99.89-1.99 2L2 19c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2zm-6 0h-4V4h4v2z"
ICO_REPORT = "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"
ICO_ALERT = "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"
ICO_GITHUB = "M12 2A10 10 0 0 0 2 12c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.1-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"
ICO_SHIELD = "M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"

@st.cache_data
def load_student_data():
    for path in ['data/student_placement_data_v2.csv', 'data/student_placement_data.csv', 'student_placement_data.csv']:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("Could not locate student placement dataset.")

df_students = load_student_data()

# ----------------- MODERN EXECUTIVE SAAS STYLING -----------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

*, html, body, [class*="css"], [class*="st-"], .stMarkdown, p, span, label, input, button, select, div {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

code, pre, .mono {{
    font-family: 'JetBrains Mono', monospace !important;
}}

.stApp {{
    background-color: #080c16 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(6, 182, 212, 0.08) 0px, transparent 45%),
        radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.09) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%) !important;
    background-attachment: fixed !important;
    color: #f1f5f9 !important;
}}

/* Top Navigation Bar */
.navbar-wrapper {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(13, 20, 36, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px 24px;
    margin-bottom: 22px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(16px);
}}

.nav-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
}}

.nav-title {{
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.3px;
}}

.nav-pill {{
    background: rgba(6, 182, 212, 0.12);
    border: 1px solid rgba(6, 182, 212, 0.3);
    color: #22d3ee;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

/* Clean Pill Badge for GitHub Repo */
.github-badge {{
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 9999px !important;
    padding: 6px 14px !important;
    color: #cbd5e1 !important;
    text-decoration: none !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}}

.github-badge, .github-badge * {{
    text-decoration: none !important;
}}

.github-badge:hover {{
    color: #38bdf8 !important;
    border-color: rgba(56, 189, 248, 0.45) !important;
    background: rgba(56, 189, 248, 0.1) !important;
    box-shadow: 0 0 14px rgba(56, 189, 248, 0.2) !important;
    transform: translateY(-1px) !important;
}}

.github-badge svg {{
    transition: transform 0.2s ease !important;
    fill: #94a3b8 !important;
}}

.github-badge:hover svg {{
    fill: #38bdf8 !important;
    transform: scale(1.08) !important;
}}

/* Clean Custom Input Elements */
[data-testid="InputInstructions"] {{
    display: none !important;
}}

div[data-baseweb="input"], 
div[data-baseweb="select"] > div {{
    background: #0d1527 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    transition: all 0.2s ease !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.5) !important;
}}

div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within {{
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.2) !important;
}}

input[type="text"], input[type="number"] {{
    color: #38bdf8 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}}

button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"] {{
    background: rgba(15, 23, 42, 0.6) !important;
    color: #94a3b8 !important;
    border: none !important;
    border-left: 1px solid rgba(255, 255, 255, 0.08) !important;
}}

button[data-testid="stNumberInputStepDown"]:hover,
button[data-testid="stNumberInputStepUp"]:hover {{
    background: rgba(6, 182, 212, 0.15) !important;
    color: #06b6d4 !important;
}}

div[data-testid="stWidgetLabel"] label, 
div[data-testid="stWidgetLabel"] p {{
    color: #cbd5e1 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    margin-bottom: 6px !important;
}}

/* Hero Card */
.hero-card {{
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(20, 27, 50, 0.7) 50%, rgba(13, 20, 36, 0.9) 100%);
    border: 1px solid rgba(6, 182, 212, 0.25);
    border-radius: 16px;
    padding: 38px;
    margin: 8px 0 24px 0;
    text-align: center;
}}

.hero-title {{
    font-size: 2.5rem;
    font-weight: 800;
    margin: 12px 0 8px 0;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #ffffff 0%, #38bdf8 45%, #a855f7 90%, #ffffff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.stat-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 24px 0;
}}

.stat-card {{
    background: rgba(13, 20, 36, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}}

.stat-card:hover {{
    transform: translateY(-2px);
    border-color: rgba(6, 182, 212, 0.4);
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.12);
}}

.feature-box {{
    background: rgba(13, 20, 36, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 20px;
    height: 100%;
}}

div[data-testid="stForm"] {{
    background: rgba(13, 20, 36, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 26px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45) !important;
}}

div[data-testid="stFormSubmitButton"] > button {{
    background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(6, 182, 212, 0.35) !important;
    transition: all 0.25s ease !important;
}}

div[data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(6, 182, 212, 0.55) !important;
}}

/* Pillar Badges */
.pill-badge {{
    padding: 4px 10px;
    border-radius: 5px;
    font-size: 0.78rem;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}}
.badge-lx {{ background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); color: #34d399; }}
.badge-ax {{ background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35); color: #38bdf8; }}
.badge-cx {{ background: rgba(168, 85, 247, 0.18); border: 1px solid rgba(168, 85, 247, 0.35); color: #c084fc; }}
.badge-px {{ background: rgba(244, 63, 94, 0.18); border: 1px solid rgba(244, 63, 94, 0.35); color: #fb7185; }}
.badge-sx {{ background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.35); color: #fbbf24; }}

.card-placed {{
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(13, 22, 38, 0.9) 100%);
    border-left: 5px solid #10b981;
    border-top: 1px solid rgba(16, 185, 129, 0.3);
    border-right: 1px solid rgba(16, 185, 129, 0.15);
    border-bottom: 1px solid rgba(16, 185, 129, 0.15);
    border-radius: 0 10px 10px 0;
    padding: 22px;
}}

.card-unplaced {{
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(20, 10, 20, 0.9) 100%);
    border-left: 5px solid #ef4444;
    border-top: 1px solid rgba(239, 68, 68, 0.3);
    border-right: 1px solid rgba(239, 68, 68, 0.15);
    border-bottom: 1px solid rgba(239, 68, 68, 0.15);
    border-radius: 0 10px 10px 0;
    padding: 22px;
}}

.card-warning {{
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(20, 16, 10, 0.9) 100%);
    border-left: 5px solid #f59e0b;
    border-top: 1px solid rgba(245, 158, 11, 0.3);
    border-right: 1px solid rgba(245, 158, 11, 0.15);
    border-bottom: 1px solid rgba(245, 158, 11, 0.15);
    border-radius: 0 10px 10px 0;
    padding: 22px;
}}

.metric-tier-card {{
    background: rgba(13, 20, 36, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 22px;
}}

.section-banner-cyan {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(6, 182, 212, 0.08);
    border: 1px solid rgba(6, 182, 212, 0.2);
    border-left: 4px solid #06b6d4;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 16px 0 14px 0;
}}

.section-banner-purple {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(147, 51, 234, 0.1);
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-left: 4px solid #c084fc;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 16px 0 14px 0;
}}

.section-banner-emerald {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-left: 4px solid #10b981;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 16px 0 14px 0;
}}
</style>

<!-- TOP NAVBAR -->
<div class="navbar-wrapper">
    <div class="nav-brand">
        {svg_icon(ICO_CHIP, '#06b6d4', 22)}
        <span class="nav-title">NEXUS // CAREERPULSE AI</span>
        <span class="nav-pill">v2.6 CALIBRATED</span>
    </div>
    <a class="github-badge" href="https://github.com/stutikatiyar/placement-prediction" target="_blank" rel="noopener noreferrer">
        {svg_icon(ICO_GITHUB, '#cbd5e1', 16)}
        <span>GitHub Repo</span>
    </a>
</div>
""", unsafe_allow_html=True)

# Clean Modern Nav Buttons
tab_col1, tab_col2, tab_col3 = st.columns([1.2, 1.6, 1.2])

with tab_col1:
    if st.button("System Overview", use_container_width=True, type="primary" if st.session_state.current_page == "overview" else "secondary"):
        st.session_state.current_page = "overview"
        st.rerun()

with tab_col2:
    if st.button("Diagnostic Evaluator", use_container_width=True, type="primary" if st.session_state.current_page == "analyze" else "secondary"):
        st.session_state.current_page = "analyze"
        st.rerun()

with tab_col3:
    if st.button("ML Benchmarks", use_container_width=True, type="primary" if st.session_state.current_page == "benchmarks" else "secondary"):
        st.session_state.current_page = "benchmarks"
        st.rerun()

st.write("")

# ==============================================================================
# PAGE 1: SYSTEM OVERVIEW
# ==============================================================================
if st.session_state.current_page == "overview":
    st.markdown(f"""
    <div class="hero-card">
        <div style="display:inline-flex; align-items:center; gap:8px; background:rgba(6,182,212,0.12); border:1px solid rgba(6,182,212,0.3); padding:4px 14px; border-radius:20px; font-size:0.75rem; color:#38bdf8; font-weight:700;">
            {svg_icon(ICO_SHIELD, '#06b6d4', 14)} RECRUITMENT INTELLIGENCE PLATFORM
        </div>
        <h1 class="hero-title">NEXUS CAREERPULSE AI</h1>
        <p style="color:#94a3b8; max-width:750px; margin:0 auto 20px auto; font-size:0.95rem; line-height:1.6;">
            A multi-modular predictive analytics platform that evaluates candidate placement probabilities across 
            academic metrics, institutional cutoffs, and 5-pillar skill assessment clearance tiers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Primary Action CTA Button
    col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
    with col_c2:
        if st.button("Launch Candidate Diagnostic", use_container_width=True):
            st.session_state.current_page = "analyze"
            st.rerun()

    st.write("")
    
    # 4-Column Metric Grid
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div style="color:#00f2fe; font-size:1.9rem; font-weight:800;">12,000+</div>
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600; margin-top:4px;">Candidate Profiles Evaluated</div>
        </div>
        <div class="stat-card">
            <div style="color:#a855f7; font-size:1.9rem; font-weight:800;">5-Pillars</div>
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600; margin-top:4px;">Modular Assessment Rubrics</div>
        </div>
        <div class="stat-card">
            <div style="color:#10b981; font-size:1.9rem; font-weight:800;">60.0%</div>
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600; margin-top:4px;">Strict Secondary Board Cutoff</div>
        </div>
        <div class="stat-card">
            <div style="color:#f43f5e; font-size:1.9rem; font-weight:800;">0 Backlog</div>
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600; margin-top:4px;">Mandatory Corporate Policy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Core Policy Modules & Diagnostic Engine")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""
        <div class="feature-box" style="border-left-color: #00f2fe;">
            <div style="font-size:1.05rem; font-weight:700; color:#38bdf8; margin-bottom:8px;">
                {svg_icon(ICO_SEARCH, '#00f2fe', 16)} Automated USN Directory Lookup
            </div>
            <p style="font-size:0.85rem; color:#94a3b8; line-height:1.6; margin:0;">
                Pre-indexed candidate records with automatic form loading. Eliminate manual input with instant lookup by name or institutional USN code.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="feature-box" style="border-left-color: #f59e0b;">
            <div style="font-size:1.05rem; font-weight:700; color:#fbbf24; margin-bottom:8px;">
                {svg_icon(ICO_ALERT, '#f59e0b', 16)} 60% Board & Backlog Screening
            </div>
            <p style="font-size:0.85rem; color:#94a3b8; line-height:1.6; margin:0;">
                Enforces corporate criteria: disqualifies candidates with active backlogs (0.0% probability) and flags severe risk for board marks under 60%.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="feature-box" style="border-left-color: #a855f7;">
            <div style="font-size:1.05rem; font-weight:700; color:#c084fc; margin-bottom:8px;">
                {svg_icon(ICO_BRIEFCASE, '#c084fc', 16)} Calibrated Package Bands
            </div>
            <p style="font-size:0.85rem; color:#94a3b8; line-height:1.6; margin:0;">
                Predicts continuous probability percentages and matches qualified candidates into tiered compensation bands (Mass IT 4-5 LPA up to Tier-1 MNC 14-20 LPA).
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 2: DIAGNOSTIC EVALUATOR
# ==============================================================================
elif st.session_state.current_page == "analyze":
    st.markdown(f"""
    <div class="section-banner-cyan">
        {svg_icon(ICO_SEARCH, '#00f2fe', 18)}
        <span style="font-size:1rem; font-weight:700; color:#f1f5f9;">Candidate Directory & Profile Auto-Population</span>
    </div>
    """, unsafe_allow_html=True)

    has_meta = ('usn' in df_students.columns and 'student_name' in df_students.columns)

    if has_meta:
        options = ['[+] Manual Profile Entry'] + [
            f"{row.usn} - {row.student_name} ({row.branch} | CGPA: {row.cgpa})"
            for _, row in df_students.head(300).iterrows()
        ]
    else:
        options = ['[+] Manual Profile Entry'] + [
            f"ID #{row.student_id:04d} ({row.branch} | CGPA: {row.cgpa})"
            for _, row in df_students.head(300).iterrows()
        ]

    selected = st.selectbox("Candidate Search:", options, label_visibility="collapsed")
    is_custom = (selected == '[+] Manual Profile Entry')

    if is_custom:
        d_name, d_usn = "", ""
        d_gender, d_age, d_degree, d_branch = "Male", 21, "BTech", "CS"
        d_tenth, d_twelfth = 82.5, 80.0
        d_cgpa, d_back, d_int, d_cert, d_proj = 8.10, 0, 1, 2, 2
        d_coding, d_comm, d_apt = 7, 7, 75
        d_lx, d_ax, d_cx, d_px, d_sx = 3, 3, 3, 3.5, 3
    else:
        if has_meta:
            sel_usn = selected.split(' - ')[0]
            row = df_students[df_students['usn'] == sel_usn].iloc[0]
            d_name, d_usn = row['student_name'], row['usn']
        else:
            sel_id = int(selected.split(' (')[0].replace('ID #', ''))
            row = df_students[df_students['student_id'] == sel_id].iloc[0]
            d_name, d_usn = f"Student #{sel_id}", f"1CR23CS{sel_id:04d}"
            
        d_gender, d_age, d_degree, d_branch = row['gender'], int(row['age']), row['degree'], row['branch']
        d_cgpa, d_back = float(row['cgpa']), int(row['backlogs'])
        d_tenth = round(min(98.0, max(52.0, d_cgpa * 9.5 + 4.0)), 1)
        d_twelfth = round(min(98.0, max(50.0, d_cgpa * 9.2 + 2.0)), 1)
        d_int, d_cert, d_proj = int(row['internships']), int(row['certifications']), int(row['projects'])
        d_coding, d_comm, d_apt = int(row['coding_skills']), int(row['communication_skills']), int(row['aptitude_score'])
        d_lx = int(row['Lx_Level_Reached'])
        d_ax = int(row['Ax_Level_Reached'])
        d_cx = int(row['Cx_Level_Reached'])
        d_px = float(row['Px_Level_Reached'])
        d_sx = int(row['Sx_Level_Reached'])

    with st.form("student_form"):
        st.markdown(f"""
        <div class="section-banner-purple">
            {svg_icon(ICO_ID, '#c084fc', 18)}
            <span style="font-size:1rem; font-weight:700; color:#f1f5f9;">Academic History & Candidate Identity</span>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            s_name = st.text_input("Full Name", value=d_name, placeholder="e.g. Aarav Sharma")
            s_usn = st.text_input("University USN", value=d_usn, placeholder="e.g. 1CR23CS0042")
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if d_gender == "Male" else 1)
            age = st.number_input("Age", 18, 30, value=d_age)
            
        with c2:
            degree = st.selectbox("Degree Program", ["BTech", "BE", "BCA", "BSc"], 
                                  index=["BTech", "BE", "BCA", "BSc"].index(d_degree) if d_degree in ["BTech", "BE", "BCA", "BSc"] else 0)
            branch = st.selectbox("Discipline / Branch", ["CS", "IT", "AI", "DS", "Electrical", "Mechanical"],
                                  index=["CS", "IT", "AI", "DS", "Electrical", "Mechanical"].index(d_branch) if d_branch in ["CS", "IT", "AI", "DS", "Electrical", "Mechanical"] else 0)
            tenth_pct = st.number_input("10th / Secondary Board (%)", 35.0, 100.0, value=float(d_tenth), step=0.5, format="%.1f")
            twelfth_pct = st.number_input("12th / PUC / Diploma (%)", 35.0, 100.0, value=float(d_twelfth), step=0.5, format="%.1f")
            
        with c3:
            cgpa = st.number_input("College Cumulative CGPA", 0.0, 10.0, value=d_cgpa, step=0.05, format="%.2f")
            backlogs = st.number_input("Active Backlogs (0 Mandatory)", 0, 10, value=d_back)
            internships = st.number_input("Internships Completed", 0, 10, value=d_int)
            certifications = st.number_input("Certifications Earned", 0, 15, value=d_cert)

        c4, c5, c6 = st.columns(3)
        with c4:
            projects = st.number_input("Capstone Projects Built", 0, 15, value=d_proj)
        with c5:
            coding_skills = st.slider("Coding Fluency Rating (1-10)", 1, 10, value=d_coding)
        with c6:
            communication_skills = st.slider("Communication Index (1-10)", 1, 10, value=d_comm)
            aptitude_score = st.slider("Aptitude Test Score (40-99)", 40, 99, value=d_apt)

        st.markdown(f"""
        <div class="section-banner-emerald">
            {svg_icon(ICO_LAYERS, '#10b981', 18)}
            <span style="font-size:1rem; font-weight:700; color:#f1f5f9;">Department Modular Cutoff Clearances (5-Pillar Rubric)</span>
        </div>
        """, unsafe_allow_html=True)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown('<span class="pill-badge badge-lx">Language (Lx)</span>', unsafe_allow_html=True)
            lx = st.selectbox("Lx Level", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(int(d_lx)), key="lx_box")
        with m2:
            st.markdown('<span class="pill-badge badge-ax">Aptitude (Ax)</span>', unsafe_allow_html=True)
            ax = st.selectbox("Ax Level", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(int(d_ax)), key="ax_box")
        with m3:
            st.markdown('<span class="pill-badge badge-cx">Core Test (Cx)</span>', unsafe_allow_html=True)
            cx = st.selectbox("Cx Level", [0, 2, 3, 4, 5], index=[0, 2, 3, 4, 5].index(int(d_cx)) if int(d_cx) in [0, 2, 3, 4, 5] else 2, key="cx_box")
        with m4:
            st.markdown('<span class="pill-badge badge-px">Programming (Px)</span>', unsafe_allow_html=True)
            px = st.selectbox("Px Level", [0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0], index=[0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0].index(float(d_px)), key="px_box")
        with m5:
            st.markdown('<span class="pill-badge badge-sx">Soft Skills (Sx)</span>', unsafe_allow_html=True)
            sx = st.selectbox("Sx Level", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(int(d_sx)), key="sx_box")

        st.write("")
        submitted = st.form_submit_button("Run Diagnostic Assessment")

    if submitted:
        final_name = s_name.strip() if s_name.strip() else ("Candidate" if is_custom else d_name)
        final_usn = s_usn.strip() if s_usn.strip() else ("1CR23CS9999" if is_custom else d_usn)
        
        student_data = pd.DataFrame([{
            'gender': gender,
            'age': age,
            'degree': degree,
            'branch': branch,
            'cgpa': cgpa,
            'backlogs': backlogs,
            'internships': internships,
            'certifications': certifications,
            'coding_skills': coding_skills,
            'communication_skills': communication_skills,
            'aptitude_score': aptitude_score,
            'projects': projects,
            'Lx_Level_Reached': lx,
            'Ax_Level_Reached': ax,
            'Cx_Level_Reached': cx,
            'Px_Level_Reached': px,
            'Sx_Level_Reached': sx
        }])
        
        pred = pipeline.predict(student_data)[0]
        raw_prob = pipeline.predict_proba(student_data)[0][1] * 100
        overall = min(lx, ax, cx, px, sx)

        has_backlogs = (backlogs > 0)
        board_below_60 = (tenth_pct < 60.0) or (twelfth_pct < 60.0)
        cgpa_below_6 = (cgpa < 6.0)
        cgpa_modifier = (cgpa - 7.8) * 8.0

        if has_backlogs:
            placed_prob = 0.0
            status_placed = False
            rejection_reason = "ACTIVE_BACKLOGS"
        elif board_below_60:
            placed_prob = np.clip(12.0 - (60.0 - min(tenth_pct, twelfth_pct)) * 0.8, 2.0, 14.0)
            status_placed = False
            rejection_reason = "BOARD_MARKS_BELOW_60"
        elif cgpa_below_6:
            placed_prob = np.clip(10.0 + (cgpa - 5.0) * 8.0, 3.0, 18.0)
            status_placed = False
            rejection_reason = "CGPA_BELOW_FIRST_CLASS"
        elif pred == 1:
            placed_prob = np.clip(raw_prob + cgpa_modifier + (min(tenth_pct, twelfth_pct) - 75.0) * 0.2, 52.0, 98.5)
            if cgpa < 7.0:
                placed_prob = min(placed_prob, 64.0)
            elif cgpa < 7.5:
                placed_prob = min(placed_prob, 76.0)
                
            status_placed = True
            rejection_reason = None
        else:
            placed_prob = np.clip(raw_prob * 0.4 + (overall / 3.0) * 10.0 + (cgpa / 10.0) * 8.0, 3.0, 42.0)
            status_placed = False
            rejection_reason = "TIER_CUTOFF_FAILED"

        st.markdown(f"""
        <div class="section-banner-cyan">
            {svg_icon(ICO_REPORT, '#00f2fe', 18)}
            <span style="font-size:1rem; font-weight:700; color:#f1f5f9;">Diagnostic Placement & Package Output</span>
        </div>
        """, unsafe_allow_html=True)
        
        col_res, col_metrics = st.columns([1.4, 1.0])
        
        with col_res:
            if rejection_reason == "ACTIVE_BACKLOGS":
                st.markdown(f"""
                <div class="card-unplaced">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h2 style="color: #ef4444; margin: 0; font-weight:800; font-size:1.25rem;">
                            {svg_icon(ICO_BAN, '#ef4444', 20)} Ineligible: Active Backlogs Detected
                        </h2>
                        <span style="background: rgba(239, 68, 68, 0.25); border: 1px solid rgba(248, 113, 113, 0.5); color: #fca5a5; padding: 4px 14px; border-radius: 4px; font-weight:700; font-size:0.85rem;">
                            0.0% Eligibility
                        </span>
                    </div>
                    <p style="color: #fca5a5; margin: 10px 0 12px 0; font-weight:600;">
                        {final_name} [{final_usn}] — {backlogs} Active Backlog(s)
                    </p>
                    <div style="background: rgba(10, 14, 26, 0.85); padding: 14px 18px; border-radius: 6px; border: 1px solid rgba(239, 68, 68, 0.35);">
                        <div style="font-size:0.8rem; text-transform:uppercase; color:#fda4af; font-weight:700;">Mandatory Corporate Policy Filter</div>
                        <div style="font-size: 0.88rem; color: #fecdd3; margin-top: 4px; line-height: 1.6;">
                            Even if assessment tiers are cleared (Level {overall}), campus recruitment portals automatically filter out candidates with active backlogs. Clear backlog to unlock drive registration.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif rejection_reason == "BOARD_MARKS_BELOW_60":
                failed_board = "10th Grade" if tenth_pct < 60.0 else "12th Grade"
                st.markdown(f"""
                <div class="card-warning">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h2 style="color: #f59e0b; margin: 0; font-weight:800; font-size:1.25rem;">
                            {svg_icon(ICO_ALERT, '#f59e0b', 20)} Critical Risk: Board Criteria Unmet
                        </h2>
                        <span style="background: rgba(245, 158, 11, 0.25); border: 1px solid rgba(251, 191, 36, 0.5); color: #fde68a; padding: 4px 14px; border-radius: 4px; font-weight:700; font-size:0.85rem;">
                            {placed_prob:.1f}% Critical Risk
                        </span>
                    </div>
                    <p style="color: #fde68a; margin: 10px 0 12px 0; font-weight:600;">
                        {final_name} [{final_usn}] — 10th: {tenth_pct}% | 12th: {twelfth_pct}%
                    </p>
                    <div style="background: rgba(10, 14, 26, 0.85); padding: 14px 18px; border-radius: 6px; border: 1px solid rgba(245, 158, 11, 0.35);">
                        <div style="font-size:0.8rem; text-transform:uppercase; color:#fbbf24; font-weight:700;">Corporate 60% First Class Barrier</div>
                        <div style="font-size: 0.88rem; color: #fef3c7; margin-top: 4px; line-height: 1.6;">
                            Over 90% of campus hiring companies enforce a <b>strict 60% aggregate cutoff in 10th and 12th</b>. Because your score in {failed_board} is below 60%, corporate ATS filters will reject the candidate before the assessment round.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif status_placed:
                if overall == 3.0:
                    pkg, tier_title = "4.0 - 5.0 LPA", "Mass Recruiter / IT Services"
                elif overall == 3.5:
                    pkg, tier_title = "5.0 - 8.0 LPA", "Digital Specialist / Consulting"
                elif overall == 4.0:
                    pkg = "8.0 - 12.0 LPA" if cgpa < 7.8 else "8.0 - 14.0 LPA"
                    tier_title = "Mid-tier Product / FinTech"
                else:
                    pkg = "12.0 - 16.0 LPA" if cgpa < 8.0 else "14.0 - 20.0 LPA"
                    tier_title = "Tier-1 Tech MNC / AI Labs"

                if cgpa >= 8.5:
                    cgpa_note = "High CGPA (> 8.5) unlocks 100% of visiting campus companies."
                elif cgpa >= 7.5:
                    cgpa_note = "CGPA between 7.5 and 8.5 satisfies cutoffs for ~85% of tier-1 & tier-2 firms."
                elif cgpa >= 7.0:
                    cgpa_note = "CGPA in 7.0-7.4 band reduces probability. Filtered out of high-paying companies requiring 7.5+ cutoff."
                else:
                    cgpa_note = "CGPA in 6.0-6.9 band is near the 60% boundary. Heavy competition in mass drives."

                st.markdown(f"""
                <div class="card-placed">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h2 style="color: #10b981; margin: 0; font-weight:800; font-size:1.25rem;">
                            {svg_icon(ICO_CHECK, '#10b981', 20)} Status: High Placement Likelihood
                        </h2>
                        <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.5); color: #6ee7b7; padding: 4px 14px; border-radius: 4px; font-weight:700; font-size:0.85rem;">
                            {placed_prob:.1f}% Probability
                        </span>
                    </div>
                    <p style="color: #e2e8f0; margin: 10px 0 16px 0;">
                        Candidate <b>{final_name}</b> [{final_usn}] satisfies all test tiers, zero-backlog policy, and secondary board cutoffs.
                    </p>
                    <div style="background: rgba(10, 14, 26, 0.8); padding: 14px 18px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.25);">
                        <div style="font-size:0.75rem; text-transform:uppercase; color:#94a3b8; font-weight:700;">
                            Projected Compensation Band
                        </div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: #f8fafc; margin: 2px 0;">{pkg}</div>
                        <div style="font-size: 0.85rem; color: #38bdf8;">Target Tier: <b>{tier_title}</b></div>
                        <div style="margin-top: 8px; font-size: 0.8rem; color: #64748b; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px;">
                            {cgpa_note}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card-unplaced">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h2 style="color: #ef4444; margin: 0; font-weight:800; font-size:1.25rem;">
                            {svg_icon(ICO_BAN, '#ef4444', 20)} Status: Below Level 3 Cutoff
                        </h2>
                        <span style="background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; padding: 4px 14px; border-radius: 4px; font-weight:700; font-size:0.85rem;">
                            {100 - placed_prob:.1f}% Risk
                        </span>
                    </div>
                    <p style="color: #e2e8f0; margin: 10px 0 0 0;">
                        Candidate <b>{final_name}</b> [{final_usn}] has an overall score below the Level 3 cutoff required for corporate placement drives.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with col_metrics:
            st.markdown(f"""
            <div class="metric-tier-card">
                <div style="font-size:0.8rem; text-transform:uppercase; color:#38bdf8; font-weight:700;">
                    {svg_icon(ICO_LAYERS, '#38bdf8', 15)} Assessment Tier Vector
                </div>
                <h1 style="color: #f8fafc; margin: 4px 0 12px 0; font-size: 2.2rem; font-weight:800;">LEVEL {overall}</h1>
                <div style="display:flex; flex-direction:column; gap:8px; font-size:0.88rem;">
                    <div style="display:flex; justify-content:space-between;"><span style="color:#94a3b8;">Language (Lx):</span> <b>L{lx} / 4</b></div>
                    <div style="display:flex; justify-content:space-between;"><span style="color:#94a3b8;">Aptitude (Ax):</span> <b>L{ax} / 4</b></div>
                    <div style="display:flex; justify-content:space-between;"><span style="color:#94a3b8;">Core Test (Cx):</span> <b>L{cx} / 5</b></div>
                    <div style="display:flex; justify-content:space-between;"><span style="color:#94a3b8;">Programming (Px):</span> <b>L{px} / 5</b></div>
                    <div style="display:flex; justify-content:space-between;"><span style="color:#94a3b8;">Soft Skills (Sx):</span> <b>L{sx} / 4</b></div>
                </div>
                <div style="margin-top:14px; padding-top:10px; border-top:1px solid rgba(255,255,255,0.08); font-size:0.8rem; color:#94a3b8;">
                    <div>10th Board: <b style="color:#ffffff;">{tenth_pct}%</b></div>
                    <div>12th Board: <b style="color:#ffffff;">{twelfth_pct}%</b></div>
                    <div>College CGPA: <b style="color:#ffffff;">{cgpa}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 3: MODEL BENCHMARK REPORT
# ==============================================================================
elif st.session_state.current_page == "benchmarks":
    st.markdown(f"""
    <div class="section-banner-purple">
        {svg_icon(ICO_REPORT, '#c084fc', 18)}
        <span style="font-size:1rem; font-weight:700; color:#f1f5f9;">Machine Learning Architecture & Performance Evaluation</span>
    </div>
    """, unsafe_allow_html=True)

    # Load dynamic evaluation benchmark metrics if available
    benchmark_file = 'models/benchmark_results.json'
    if os.path.exists(benchmark_file):
        df_bench = pd.read_json(benchmark_file)
        df_bench['Accuracy'] = df_bench['Accuracy'].apply(lambda x: f"{x * 100:.2f}%")
        df_bench['Precision'] = df_bench['Precision'].apply(lambda x: f"{x * 100:.2f}%")
        df_bench['Recall'] = df_bench['Recall'].apply(lambda x: f"{x * 100:.2f}%")
        df_bench['F1-Score'] = df_bench['F1-Score'].apply(lambda x: f"{x:.4f}")
        df_bench['Status'] = df_bench['Model'].apply(
            lambda m: "Active Deployment" if "Random Forest" in m else "Evaluated"
        )
        benchmark_data = df_bench.rename(columns={"Model": "Model Architecture"})
    else:
        benchmark_data = pd.DataFrame([
            {"Model Architecture": "Random Forest (Calibrated)", "Accuracy": "76.33%", "Precision": "77.48%", "Recall": "82.05%", "F1-Score": "0.7970", "Status": "Active Deployment"},
            {"Model Architecture": "Gradient Boosting (GBM)", "Accuracy": "76.17%", "Precision": "77.38%", "Recall": "81.82%", "F1-Score": "0.7954", "Status": "Evaluated"},
            {"Model Architecture": "Logistic Regression", "Accuracy": "74.88%", "Precision": "76.00%", "Recall": "81.31%", "F1-Score": "0.7856", "Status": "Evaluated"},
            {"Model Architecture": "K-Nearest Neighbors (KNN)", "Accuracy": "71.42%", "Precision": "72.63%", "Recall": "79.47%", "F1-Score": "0.7590", "Status": "Baseline"}
        ])

    st.dataframe(benchmark_data, use_container_width=True, hide_index=True)

    # Visualizations: Confusion Matrix and Feature Importance
    st.markdown("### 📊 Diagnostic Visualizations & Interpretability")
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        if os.path.exists('models/confusion_matrix.png'):
            st.image('models/confusion_matrix.png', caption="Confusion Matrix: Random Forest (Test Set N=2,400)", use_container_width=True)
        else:
            st.info("Run train.py to generate the confusion matrix plot.")

    with v_col2:
        if os.path.exists('models/feature_importance.png'):
            st.image('models/feature_importance.png', caption="Top 10 Feature Importances (Random Forest)", use_container_width=True)
        else:
            st.info("Run train.py to generate feature importances.")

    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"""
        <div style="background: rgba(13,20,36,0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 22px; height: 100%;">
            <div style="font-size:1.05rem; font-weight:700; color:#38bdf8; margin-bottom:8px;">
                {svg_icon(ICO_CHIP, '#38bdf8', 16)} Ensemble Calibration Architecture
            </div>
            <p style="color:#94a3b8; font-size:0.88rem; line-height:1.6; margin-bottom:0;">
                A standard decision tree yields rigid binary step probabilities (0.0 or 1.0). The active production pipeline utilizes an ensemble of 150 regularized decision estimators with leaf smoothing (<code>min_samples_leaf=15</code>). This provides calibrated, continuous confidence scores directly tied to candidate CGPA distributions and test clearances.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown(f"""
        <div style="background: rgba(13,20,36,0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 22px; height: 100%;">
            <div style="font-size:1.05rem; font-weight:700; color:#34d399; margin-bottom:8px;">
                {svg_icon(ICO_CHECK, '#34d399', 16)} Strict Data Leakage Prevention
            </div>
            <p style="color:#94a3b8; font-size:0.88rem; line-height:1.6; margin-bottom:0;">
                All non-predictive student identifiers (<code>student_id</code>, <code>student_name</code>, <code>usn</code>) and target composite scores (<code>Overall_Level_Score</code>) are strictly dropped prior to feature transformation. Standard scaling and one-hot encoding are fit exclusively inside the Scikit-Learn pipeline to avoid train-test contamination.
            </p>
        </div>
        """, unsafe_allow_html=True)
