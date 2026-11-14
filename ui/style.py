"""
style.py - look and motion of the MiniRocket Explorer (CSS + small HTML building blocks)

Author : Shreyas Shinde (Student ID 26104692)

Everything visual that is not a Streamlit widget lives here, so the pages stay about content:

  inject()            the style sheet (called once per run from app.py)
  hero(...)           the animated banner on the home page
  page_header(...)    icon badge + title + one-line summary at the top of every page
  race_bars(...)      horizontal bars that grow at different speeds (time / size comparisons)
  status_chip(...)    small coloured label (used by the live early-classification replay)

The colours are the project palette (same as ui/web/index.html). The style sheet only adds
to Streamlit's theme, so the app stays readable in the light and the dark theme; the dark
theme in .streamlit/config.toml is the intended one. Motion is switched off for visitors
whose system asks for reduced motion.

HTML is written on single lines on purpose: st.markdown would turn indented or blank lines
inside HTML into Markdown code blocks.
"""
import html
import math

import streamlit as st

ROCKET, AI, EARLY, HEAT, MUTED = "#FF5C00", "#8B5CF6", "#10B981", "#EC4899", "#A1A1AA"

# One accent pair per menu section (badge gradient of the page header and the home cards)
SECTION_COLOURS = {
    "Overview": (ROCKET, HEAT),
    "Classify & explain": (AI, HEAT),
    "Real time": (EARLY, "#22D3EE"),
    "Speed & size": (ROCKET, "#FBBF24"),
    "Evidence": ("#38BDF8", AI),
}

CSS = """
<style>
:root {
  --mr-rocket: #FF5C00; --mr-ai: #8B5CF6; --mr-early: #10B981; --mr-heat: #EC4899;
  --mr-line: rgba(161, 161, 170, .28); --mr-surface: rgba(161, 161, 170, .07); --mr-muted: #A1A1AA;
}
@keyframes mr-rise  { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes mr-sheen { from { background-position: 0% 50%; } to { background-position: 200% 50%; } }
@keyframes mr-draw  { to { stroke-dashoffset: 0; } }
@keyframes mr-float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-5px); } }
@keyframes mr-pulse { 0%, 100% { opacity: .5; transform: scale(1); } 50% { opacity: 1; transform: scale(1.5); } }
@keyframes mr-grow  { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes mr-drift { from { transform: translateX(0); } to { transform: translateX(-50%); } }

/* ---- page background: two soft glows on top of the theme colour ---- */
[data-testid="stAppViewContainer"] {
  background-image:
    radial-gradient(900px 520px at 88% -8%, rgba(139, 92, 246, .16), transparent 62%),
    radial-gradient(760px 460px at -6% 12%, rgba(255, 92, 0, .11), transparent 60%);
  background-attachment: fixed;
}

/* ---- hero banner (home) ---- */
.mr-hero { position: relative; overflow: hidden; border-radius: 22px; padding: 34px 36px 112px;
  margin: 0 0 22px; color: #FAFAFA; border: 1px solid rgba(255, 255, 255, .10);
  background: linear-gradient(135deg, #1B1033 0%, #0F172A 48%, #2A1206 100%);
  box-shadow: 0 30px 80px -40px rgba(139, 92, 246, .55); animation: mr-rise .6s ease both; }
.mr-hero-title { font-size: clamp(30px, 4.2vw, 50px); font-weight: 800; letter-spacing: -.02em; line-height: 1.08;
  background: linear-gradient(90deg, #FFFFFF, #FFB38A, #C4B5FD, #6EE7B7, #FFFFFF); background-size: 200% 100%;
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;
  animation: mr-sheen 9s linear infinite; }
.mr-hero-sub { margin-top: 10px; max-width: 46rem; font-size: 16.5px; line-height: 1.55; color: rgba(250, 250, 250, .78); }
.mr-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; position: relative; z-index: 1; }
.mr-chip { font-size: 12.5px; padding: 6px 12px; border-radius: 999px; color: #FAFAFA;
  background: rgba(255, 255, 255, .08); border: 1px solid rgba(255, 255, 255, .16);
  animation: mr-rise .6s ease both; }
.mr-chip b { color: #FFB38A; font-weight: 700; }
.mr-wave { position: absolute; left: 0; right: 0; bottom: 8px; width: 100%; height: 92px; opacity: .95; pointer-events: none; }
.mr-wave path { fill: none; stroke-linecap: round; stroke-linejoin: round; }
.mr-wave .mr-wave-line { stroke: url(#mr-stroke); stroke-width: 2.4; stroke-dasharray: 1; stroke-dashoffset: 1;
  animation: mr-draw 3.2s cubic-bezier(.45, 0, .2, 1) .25s forwards; }
.mr-wave .mr-wave-ghost { stroke: rgba(255, 255, 255, .10); stroke-width: 1.2; }
.mr-wave circle { fill: #6EE7B7; transform-box: fill-box; transform-origin: center; opacity: 0;
  animation: mr-rise .4s ease 3.3s forwards, mr-pulse 1.8s ease-in-out 3.7s infinite; }
.mr-hero > :not(.mr-wave) { position: relative; z-index: 1; }

/* ---- page header (every feature page) ---- */
.mr-header { display: flex; align-items: center; gap: 16px; margin: 2px 0 6px; animation: mr-rise .45s ease both; }
.mr-badge { flex: 0 0 auto; width: 58px; height: 58px; border-radius: 16px; display: grid; place-items: center;
  font-size: 28px; background: linear-gradient(135deg, var(--c1), var(--c2));
  box-shadow: 0 14px 34px -14px var(--c1); animation: mr-float 4.5s ease-in-out infinite; }
.mr-title { font-size: clamp(26px, 3vw, 36px); font-weight: 800; letter-spacing: -.02em; line-height: 1.15; }
.mr-sub { margin-top: 3px; font-size: 15px; color: var(--mr-muted); }
.mr-rule { height: 3px; border-radius: 3px; margin: 12px 0 18px; transform-origin: left;
  background: linear-gradient(90deg, var(--c1), var(--c2), transparent 85%);
  animation: mr-grow .9s cubic-bezier(.2, .8, .2, 1) .1s both; }

/* ---- race bars ---- */
.mr-race { display: grid; gap: 10px; margin: 6px 0 14px; }
.mr-race-row { display: grid; grid-template-columns: minmax(120px, 190px) 1fr minmax(70px, auto); gap: 12px; align-items: center; font-size: 14px; }
.mr-race-row span { color: var(--mr-muted); }
.mr-race-row b { text-align: right; font-variant-numeric: tabular-nums; }
.mr-track { height: 14px; border-radius: 999px; background: var(--mr-surface); border: 1px solid var(--mr-line); overflow: hidden; }
.mr-bar { height: 100%; width: var(--w); border-radius: 999px; transform-origin: left;
  background: linear-gradient(90deg, var(--c1), var(--c2));
  animation: mr-grow var(--t, 1s) cubic-bezier(.3, .7, .2, 1) both; }

/* ---- status chip ---- */
.mr-status { display: inline-flex; align-items: center; gap: 8px; padding: 7px 14px; border-radius: 999px;
  font-size: 14px; border: 1px solid var(--c); background: color-mix(in srgb, var(--c) 14%, transparent); }
.mr-status i { width: 9px; height: 9px; border-radius: 50%; background: var(--c); animation: mr-pulse 1.2s ease-in-out infinite; }

/* ---- Streamlit elements ---- */
[data-testid="stMetric"] { background: var(--mr-surface); border: 1px solid var(--mr-line); border-radius: 16px;
  padding: 14px 18px; animation: mr-rise .5s ease both;
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: var(--mr-rocket);
  box-shadow: 0 16px 36px -18px rgba(255, 92, 0, .6); }
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"], [data-testid="column"]:nth-child(2) [data-testid="stMetric"] { animation-delay: .07s; }
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"], [data-testid="column"]:nth-child(3) [data-testid="stMetric"] { animation-delay: .14s; }
[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"], [data-testid="column"]:nth-child(4) [data-testid="stMetric"] { animation-delay: .21s; }

/* feature cards on the home page (containers with a key; plain bordered boxes on older Streamlit) */
[class*="st-key-mr_card_"] {
  border-radius: 18px !important; background: var(--mr-surface); animation: mr-rise .55s ease both;
  transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; }
[class*="st-key-mr_card_"]:hover { transform: translateY(-5px); border-color: var(--mr-ai) !important;
  box-shadow: 0 22px 44px -22px rgba(139, 92, 246, .75); }
[data-testid="stColumn"]:nth-child(2) [class*="st-key-mr_card_"], [data-testid="column"]:nth-child(2) [class*="st-key-mr_card_"] { animation-delay: .08s; }
[data-testid="stColumn"]:nth-child(3) [class*="st-key-mr_card_"], [data-testid="column"]:nth-child(3) [class*="st-key-mr_card_"] { animation-delay: .16s; }
.mr-card-head { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 17px; }
.mr-card-icon { width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center; font-size: 18px;
  background: linear-gradient(135deg, var(--c1), var(--c2)); }
.mr-card-number { margin-top: 2px; font-size: 13px; font-weight: 600;
  background: linear-gradient(90deg, var(--c1), var(--c2)); -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent; }
.mr-section { margin: 20px 0 8px; font-size: 12.5px; letter-spacing: .14em; text-transform: uppercase; color: var(--mr-muted); }

[data-testid="stPageLink"] a { border-radius: 999px; border: 1px solid var(--mr-line);
  transition: transform .15s ease, border-color .15s ease, background .15s ease; }
[data-testid="stPageLink"] a:hover { transform: translateX(4px); border-color: var(--mr-rocket); }

[data-testid="stSidebar"] { border-right: 1px solid var(--mr-line); }
[data-testid="stSidebarNavLink"] { border-radius: 10px; transition: transform .15s ease, background .15s ease; }
[data-testid="stSidebarNavLink"]:hover { transform: translateX(4px); }
[data-testid="stSidebarNavLink"][aria-current="page"] { box-shadow: inset 3px 0 0 var(--mr-rocket); }

.stButton > button { border-radius: 999px; transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
.stButton > button:hover { transform: translateY(-2px); border-color: var(--mr-rocket);
  box-shadow: 0 12px 26px -14px rgba(255, 92, 0, .8); }
.stButton > button:active { transform: translateY(0); }

[data-baseweb="tab"] { transition: color .15s ease; }
[data-baseweb="tab-highlight"] { height: 3px; border-radius: 3px; background: linear-gradient(90deg, var(--mr-rocket), var(--mr-ai)); }

[data-testid="stImage"] img { border-radius: 16px; border: 1px solid var(--mr-line); animation: mr-rise .5s ease both; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 14px; overflow: hidden; animation: mr-rise .5s ease both; }
[data-testid="stAlert"] { border-radius: 14px; animation: mr-rise .4s ease both; }
[data-testid="stExpander"] details { border-radius: 14px; }

@media (prefers-reduced-motion: reduce) {
  .mr-hero, .mr-hero *, .mr-header, .mr-badge, .mr-rule, .mr-bar, .mr-status i, .mr-chip,
  [data-testid="stMetric"], [class*="st-key-mr_card_"], [data-testid="stImage"] img,
  [data-testid="stDataFrame"], [data-testid="stAlert"] { animation: none !important; transition: none !important; }
  .mr-wave .mr-wave-line { stroke-dashoffset: 0; }
  .mr-wave circle { opacity: 1; }
}
</style>
"""


def inject():
    """Add the style sheet to the page. Call once per run, before the page content."""
    st.markdown(CSS, unsafe_allow_html=True)


def _wave_path(width=1000, height=220, points=160):
    """An SVG path that looks like a time series: calm, then a burst, then calm (drawn left to right)."""
    coordinates = []
    for i in range(points + 1):
        t = i / points
        burst = math.exp(-((t - 0.56) / 0.11) ** 2)                 # where the 'event' happens
        y = (0.50
             - 0.07 * math.sin(2 * math.pi * 3.0 * t)
             - 0.05 * math.sin(2 * math.pi * 7.3 * t + 1.1)
             - 0.34 * burst * math.sin(2 * math.pi * 9.0 * t))
        coordinates.append((t * width, (0.18 + 0.72 * y) * height))
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
    return path, coordinates[-1]


def hero(title, subtitle, chips):
    """Banner for the home page. chips = list of (bold part, rest), e.g. ("84", "fixed kernels")."""
    path, (end_x, end_y) = _wave_path()
    chip_html = "".join(
        f'<span class="mr-chip" style="animation-delay:{0.35 + 0.08 * i:.2f}s"><b>{html.escape(bold)}</b> {html.escape(rest)}</span>'
        for i, (bold, rest) in enumerate(chips))
    wave = ('<svg class="mr-wave" viewBox="0 0 1000 220" preserveAspectRatio="none" aria-hidden="true">'
            '<defs><linearGradient id="mr-stroke" x1="0" x2="1" y1="0" y2="0">'
            f'<stop offset="0" stop-color="{ROCKET}"/><stop offset=".5" stop-color="{HEAT}"/>'
            f'<stop offset="1" stop-color="{EARLY}"/></linearGradient></defs>'
            f'<path class="mr-wave-ghost" d="{path}"/>'
            f'<path class="mr-wave-line" pathLength="1" d="{path}"/>'
            f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="5"/></svg>')
    st.markdown(f'<div class="mr-hero">{wave}<div class="mr-hero-title">{html.escape(title)}</div>'
                f'<div class="mr-hero-sub">{html.escape(subtitle)}</div>'
                f'<div class="mr-chips">{chip_html}</div></div>', unsafe_allow_html=True)


def page_header(icon, title, subtitle="", section="Overview"):
    """Animated header of a feature page: icon badge, title, one-line summary and a coloured rule."""
    c1, c2 = SECTION_COLOURS.get(section, SECTION_COLOURS["Overview"])
    sub = f'<div class="mr-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="mr-header" style="--c1:{c1};--c2:{c2}"><div class="mr-badge">{icon}</div>'
                f'<div><div class="mr-title">{html.escape(title)}</div>{sub}</div></div>'
                f'<div class="mr-rule" style="--c1:{c1};--c2:{c2}"></div>', unsafe_allow_html=True)


def card_head(icon, title, headline, section):
    """Icon + title + headline number inside a home-page feature card."""
    c1, c2 = SECTION_COLOURS.get(section, SECTION_COLOURS["Overview"])
    st.markdown(f'<div style="--c1:{c1};--c2:{c2}"><div class="mr-card-head"><div class="mr-card-icon">{icon}</div>'
                f'{html.escape(title)}</div><div class="mr-card-number">{html.escape(headline)}</div></div>',
                unsafe_allow_html=True)


def section_label(text):
    st.markdown(f'<div class="mr-section">{html.escape(text)}</div>', unsafe_allow_html=True)


def race_bars(rows, slowest_seconds=1.6):
    """
    Bars that grow from the left; a smaller value finishes sooner, like a race.
    rows = list of (label, value, text shown on the right, colour pair or None). Bars are scaled
    to the largest value.
    """
    largest = max((value for _, value, _, _ in rows), default=0) or 1.0
    parts = []
    for label, value, text, colours in rows:
        share = max(0.0, min(1.0, value / largest))
        c1, c2 = colours or (MUTED, MUTED)
        parts.append(f'<div class="mr-race-row"><span>{html.escape(label)}</span><div class="mr-track">'
                     f'<div class="mr-bar" style="--w:{100 * share:.1f}%;--t:{max(0.25, slowest_seconds * share):.2f}s;'
                     f'--c1:{c1};--c2:{c2}"></div></div><b>{html.escape(text)}</b></div>')
    st.markdown(f'<div class="mr-race">{"".join(parts)}</div>', unsafe_allow_html=True)


def status_chip(text, colour=MUTED):
    """HTML of a small pulsing status label (returned, so it can be written into a placeholder)."""
    return f'<span class="mr-status" style="--c:{colour}"><i></i>{html.escape(text)}</span>'
