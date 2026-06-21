"""
Global Streamlit CSS and UI component helpers.
"""

from config import THEME

# ─── Master CSS ───────────────────────────────────────────────────────────────

GLOBAL_CSS = f"""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {THEME['text_primary']};
}}

/* ── Background ── */
.stApp {{
    background: linear-gradient(135deg, #F9F6F0 0%, #F0ECE3 50%, #E8E4DB 100%);
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer {{ visibility: hidden; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {THEME['primary']} !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}}
[data-testid="stSidebar"] * {{
    color: rgba(255,255,255,0.92) !important;
}}
[data-testid="stSidebar"] .stRadio > label {{
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    padding: 0.1rem 0;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
    background: rgba(255,255,255,0.04) !important;
    border-radius: 8px;
    padding: 10px 14px !important;
    margin-bottom: 4px;
    border: 1px solid rgba(255,255,255,0.06);
    transition: background 0.15s;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.88) !important;
    cursor: pointer;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
    background: rgba(255,255,255,0.10) !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-baseweb="radio"] input:checked + div {{
    background: {THEME['accent']};
}}

/* ── Top nav bar ── */
.ser-topbar {{
    background: {THEME['surface']};
    border-bottom: 1px solid {THEME['border']};
    padding: 14px 28px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
    border-radius: 0 0 12px 12px;
}}
.ser-topbar-logo {{
    width: 36px; height: 36px;
    background: {THEME['primary']};
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
}}
.ser-topbar-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: {THEME['primary']};
    font-weight: 400;
}}
.ser-topbar-sub {{
    font-size: 0.75rem;
    color: {THEME['text_muted']};
    letter-spacing: 0.04em;
}}

/* ── Page header ── */
.ser-page-header {{
    margin-bottom: 28px;
}}
.ser-page-eyebrow {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {THEME['secondary']};
    margin-bottom: 4px;
}}
.ser-page-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: {THEME['text_primary']};
    line-height: 1.2;
    margin-bottom: 6px;
}}
.ser-page-desc {{
    font-size: 0.93rem;
    color: {THEME['text_secondary']};
    max-width: 620px;
    line-height: 1.6;
}}

/* ── KPI Cards ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}}
.kpi-card {{
    background: {THEME['surface']};
    border: 1px solid {THEME['border']};
    border-radius: 12px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s, box-shadow 0.15s;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(26,94,58,0.10);
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: {THEME['secondary']};
    border-radius: 12px 12px 0 0;
}}
.kpi-card-accent::before {{ background: {THEME['accent']}; }}
.kpi-card-warm::before {{ background: #D97706; }}
.kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {THEME['text_muted']};
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {THEME['text_primary']};
    line-height: 1;
    margin-bottom: 4px;
}}
.kpi-sub {{
    font-size: 0.78rem;
    color: {THEME['text_muted']};
}}

/* ── Section card ── */
.ser-card {{
    background: {THEME['surface']};
    border: 1px solid {THEME['border']};
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}}
.ser-card-title {{
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {THEME['text_secondary']};
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid {THEME['border']};
}}

/* ── Emotion badges ── */
.emotion-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 100px;
    padding: 6px 16px;
    font-weight: 600;
    font-size: 0.88rem;
    border: 1.5px solid;
}}

/* ── Confidence bar ── */
.conf-bar-wrap {{
    background: {THEME['surface_alt']};
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    margin-top: 6px;
}}
.conf-bar-fill {{
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, {THEME['primary']}, {THEME['accent']});
    transition: width 0.4s ease;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {{
    border-bottom: 2px solid {THEME['border']};
    gap: 0;
    background: transparent;
}}
[data-testid="stTabs"] [role="tab"] {{
    border: none !important;
    background: transparent !important;
    color: {THEME['text_secondary']} !important;
    font-weight: 500;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px;
    font-size: 0.9rem;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {THEME['primary']} !important;
    font-weight: 700 !important;
    border-bottom-color: {THEME['primary']} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {THEME['primary']} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 22px !important;
    transition: background 0.15s !important;
}}
.stButton > button:hover {{
    background: {THEME['secondary']} !important;
}}

/* ── Inputs ── */
.stTextInput input, .stNumberInput input, .stSelectbox select {{
    border: 1.5px solid {THEME['border']} !important;
    border-radius: 8px !important;
    background: {THEME['surface']} !important;
    color: {THEME['text_primary']} !important;
    font-size: 0.9rem !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
    border-color: {THEME['secondary']} !important;
    box-shadow: 0 0 0 3px rgba(45,134,83,0.12) !important;
}}

/* ── Alerts / Info boxes ── */
.ser-info {{
    background: rgba(45,134,83,0.08);
    border-left: 4px solid {THEME['secondary']};
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: {THEME['text_secondary']};
    margin-bottom: 16px;
}}
.ser-warning {{
    background: rgba(217,119,6,0.08);
    border-left: 4px solid {THEME['warning']};
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: {THEME['text_secondary']};
    margin-bottom: 16px;
}}
.ser-success {{
    background: rgba(22,163,74,0.08);
    border-left: 4px solid {THEME['success']};
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: {THEME['text_secondary']};
    margin-bottom: 16px;
}}

/* ── Dividers ── */
.ser-divider {{
    border: none;
    border-top: 1px solid {THEME['border']};
    margin: 24px 0;
}}

/* ── Footer ── */
.ser-footer {{
    text-align: center;
    padding: 28px 0 12px;
    font-size: 0.76rem;
    color: {THEME['text_muted']};
    border-top: 1px solid {THEME['border']};
    margin-top: 40px;
}}

/* ── Progress bar custom ── */
.stProgress > div > div > div {{
    background: {THEME['secondary']} !important;
}}

/* ── Tables ── */
.stDataFrame table {{
    font-size: 0.86rem !important;
}}
.stDataFrame thead tr th {{
    background: {THEME['surface_alt']} !important;
    color: {THEME['text_secondary']} !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    border: 1px solid {THEME['border']} !important;
    border-radius: 10px !important;
    background: {THEME['surface']} !important;
}}

/* ── Spinner ── */
.stSpinner > div {{
    border-top-color: {THEME['secondary']} !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {THEME['surface_alt']}; }}
::-webkit-scrollbar-thumb {{ background: {THEME['border']}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {THEME['text_muted']}; }}

/* ── Sidebar logo area ── */
.sidebar-brand {{
    padding: 20px 16px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 16px;
}}
.sidebar-brand-icon {{
    font-size: 2rem;
    margin-bottom: 6px;
}}
.sidebar-brand-name {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: white !important;
    line-height: 1.2;
}}
.sidebar-brand-tagline {{
    font-size: 0.68rem;
    color: rgba(255,255,255,0.55) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}}

/* ── Status pill ── */
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 0.74rem;
    font-weight: 600;
}}
.status-pill.online {{
    background: rgba(22,163,74,0.12);
    color: {THEME['success']};
    border: 1px solid rgba(22,163,74,0.25);
}}
.status-pill.offline {{
    background: rgba(220,38,38,0.10);
    color: {THEME['error']};
    border: 1px solid rgba(220,38,38,0.20);
}}
.status-pill.pending {{
    background: rgba(217,119,6,0.10);
    color: {THEME['warning']};
    border: 1px solid rgba(217,119,6,0.20);
}}
</style>
"""


def page_header(eyebrow: str, title: str, desc: str = "") -> str:
    desc_html = f'<p class="ser-page-desc">{desc}</p>' if desc else ""
    return f"""
    <div class="ser-page-header">
        <div class="ser-page-eyebrow">{eyebrow}</div>
        <h1 class="ser-page-title">{title}</h1>
        {desc_html}
    </div>
    """


def kpi_card(label: str, value: str, sub: str = "", variant: str = "") -> str:
    cls = f"kpi-card {f'kpi-card-{variant}' if variant else ''}"
    return f"""
    <div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {'<div class="kpi-sub">' + sub + '</div>' if sub else ''}
    </div>
    """


def section_card(title: str, content: str) -> str:
    return f"""
    <div class="ser-card">
        <div class="ser-card-title">{title}</div>
        {content}
    </div>
    """


def status_pill(text: str, state: str = "online") -> str:
    dot = "●"
    return f'<span class="status-pill {state}">{dot} {text}</span>'


def info_box(text: str, kind: str = "info") -> str:
    return f'<div class="ser-{kind}">{text}</div>'
