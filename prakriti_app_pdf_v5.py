# prakriti_app_pdf_v5.py  (consolidated latest)
# Prakriti Analyzer — By Kakunje Ayurveda

import os, io, time, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

# ---- PDF deps ----
REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle
    )
    from reportlab.lib import colors
except Exception:
    REPORTLAB_OK = False

# -------------------------
# CONFIG & PATHS
# -------------------------
st.set_page_config(page_title="Kakunje Prakriti App", layout="wide", page_icon="🪷")

BRAND_TITLE = "Kakunje Ayurveda Clinic & Research Centre"
LOGO_PATH   = Path("logo.png")                           # keep logo.png next to this file
DATA_DIR    = Path("data")
REPORTS_DIR = DATA_DIR / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV    = DATA_DIR / "prakriti_results.csv"
FEEDBACK_CSV   = DATA_DIR / "feedback.csv"
USAGE_LOG_CSV  = DATA_DIR / "usage_log.csv"
CONFIG_JSON    = DATA_DIR / "config.json"

ADMIN_PIN = st.secrets.get("ADMIN_PIN", "1234")

FOOTER_TEXT = (
    "Kakunje Ayurveda Clinic & Research Centre · Moodubidri, Karnataka, India · "
    "+91-9483697676 · kakunje.com · prasanna@kakunje.com"
)

REQUIRED_COLS = [
    "timestamp","person_key","attempt_no","name","age","gender",
    "country","state","city","consent_analytics",
    "vata_score","pitta_score","kapha_score",
    "vata_%","pitta_%","kapha_%","type"
]

# === Classification threshold defaults ===
THRESHOLD_DEFAULTS = {"TRI_GAP": 5.0, "DUAL_GAP": 7.0, "DUAL_MIN": 30.0}

# -------------------------
# QUESTIONS, WEIGHTS
# -------------------------
INT_TO_WEIGHT = {
    "Rarely": 0, "Sometimes": 1, "Often": 2,           # variable
    "Low": 0, "Moderate": 1, "High": 2,                # psychological
    "Mildly": 0, "Moderately": 1, "Strongly": 2        # physical
}

QUESTIONS = [
    ("Natural body build",
     {"Vata": "Thin/lean; hard to gain weight",
      "Pitta": "Medium/muscular; moderate weight",
      "Kapha": "Broad/heavier; gains easily"},
     "physical", 1.0),
    ("Skin baseline (without products)",
     {"Vata": "Dry/rough/cool",
      "Pitta": "Warm/sensitive; redness possible",
      "Kapha": "Smooth/thick/oily"},
     "physical", 1.0),
    ("Hair & scalp tendency",
     {"Vata": "Dry hair/scalp; frizz",
      "Pitta": "Fine hair; early greying common",
      "Kapha": "Thick/lustrous; oily scalp"},
     "physical", 0.8),
    ("Appetite pattern (day to day)",
     {"Vata": "Irregular; sometimes low/sometimes high",
      "Pitta": "Strong/sharp; hungry on time",
      "Kapha": "Slow/steady; can skip meals"},
     "variable", 1.6),
    ("Digestion / bowel pattern",
     {"Vata": "Variable; gas/bloating common",
      "Pitta": "Fast; tendency to loose/acidic",
      "Kapha": "Slower; well-formed stools"},
     "variable", 1.3),
    ("Thirst & sweating",
     {"Vata": "Low thirst; minimal sweat",
      "Pitta": "High thirst; profuse sweat",
      "Kapha": "Moderate thirst; mild sweat"},
     "variable", 1.0),
    ("Thermal comfort",
     {"Vata": "Prefers warmth; dislikes cold",
      "Pitta": "Feels hot easily; prefers cool",
      "Kapha": "Comfortable with cool; dislikes damp cold"},
     "physical", 0.9),
    ("Sleep quality",
     {"Vata": "Light; wakes easily; mind active",
      "Pitta": "Moderate; heat/dreams may disturb",
      "Kapha": "Deep/heavy; longer duration"},
     "variable", 1.3),
    ("Mental focus (typical)",
     {"Vata": "Quick ideas; distractible",
      "Pitta": "Sharp focus; task-driven",
      "Kapha": "Steady; slower but consistent"},
     "psychological", 1.5),
    ("Emotional response under stress",
     {"Vata": "Worry/anxiety; overthinking",
      "Pitta": "Irritability/anger; impatience",
      "Kapha": "Withdrawal/attachment; lethargy"},
     "psychological", 1.4),
    ("Speech & pace",
     {"Vata": "Fast/variable; animated",
      "Pitta": "Clear/precise; firm",
      "Kapha": "Slow/steady; calm"},
     "variable", 0.8),
    ("Joints & movement",
     {"Vata": "Cracking/mobility; dryness",
      "Pitta": "Warm; occasional tenderness",
      "Kapha": "Stable/cushioned; less cracking"},
     "physical", 0.9),
    ("Food preferences",
     {"Vata": "Warm, oily, grounding foods",
      "Pitta": "Cooling/light foods; dislikes very spicy",
      "Kapha": "Light/warm/spicy foods; dislikes heavy"},
     "variable", 1.0),
    ("Weather preference",
     {"Vata": "Warm/dry climate",
      "Pitta": "Cool/mild climate",
      "Kapha": "Cold/dry climate with sunshine"},
     "variable", 0.7),
    ("Typical energy pattern",
     {"Vata": "Variable energy; bursts and dips",
      "Pitta": "Strong/consistent until late crash",
      "Kapha": "Steady but slow; builds with activity"},
     "psychological", 1.2),
]

DISCLAIMER = (
    "This assessment is educational and not a medical diagnosis. "
    "Please consult a qualified Ayurveda physician for personalized advice and treatment."
)

# -------------------------
# CONFIG PERSISTENCE
# -------------------------
def load_config() -> dict:
    cfg = THRESHOLD_DEFAULTS.copy()
    if CONFIG_JSON.exists():
        try:
            on_disk = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
            for k in THRESHOLD_DEFAULTS:
                if k in on_disk:
                    cfg[k] = float(on_disk[k])
        except Exception:
            pass
    return cfg

def save_config(cfg: dict):
    try:
        CONFIG_JSON.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        st.sidebar.warning(f"Could not save config: {e}")

# seed session with config on first load
if "config" not in st.session_state:
    st.session_state.config = load_config()

# -------------------------
# SMALL HELPERS
# -------------------------
def intensity_labels_for(category: str):
    if category == "variable":
        return ["Rarely", "Sometimes", "Often"]
    if category == "psychological":
        return ["Low", "Moderate", "High"]
    return ["Mildly", "Moderately", "Strongly"]  # physical

def person_key(name: str, age: int, gender: str) -> str:
    return f"{(name or '').strip().lower()}|{int(age or 0)}|{(gender or '').strip().lower()}"

def percentify(scores: dict) -> dict:
    total = sum(scores.values()) or 1
    return {k: round(100 * v / total, 1) for k, v in scores.items()}

def get_thresholds():
    cfg = st.session_state.config or THRESHOLD_DEFAULTS
    try:
        tri = float(cfg.get("TRI_GAP", THRESHOLD_DEFAULTS["TRI_GAP"]))
        duo = float(cfg.get("DUAL_GAP", THRESHOLD_DEFAULTS["DUAL_GAP"]))
        dmn = float(cfg.get("DUAL_MIN", THRESHOLD_DEFAULTS["DUAL_MIN"]))
    except Exception:
        tri, duo, dmn = THRESHOLD_DEFAULTS["TRI_GAP"], THRESHOLD_DEFAULTS["DUAL_GAP"], THRESHOLD_DEFAULTS["DUAL_MIN"]
    return tri, duo, dmn

def prakriti_type_from_percent(perc: dict) -> str:
    """
    Decide Dominant / Dual / Tridoshic from percent dict like {"Vata": 45.0, "Pitta": 46.0, "Kapha": 9.0}.
    Uses live thresholds from config (persisted).
    """
    TRI_GAP, DUAL_GAP, DUAL_MIN = get_thresholds()

    items = sorted(perc.items(), key=lambda x: x[1], reverse=True)
    (d1, v1), (d2, v2), (d3, v3) = items

    if max(v1, v2, v3) - min(v1, v2, v3) <= TRI_GAP:
        return "Tridoshic (V-P-K)"

    if abs(v1 - v2) <= DUAL_GAP and v1 >= DUAL_MIN and v2 >= DUAL_MIN:
        pair = "".join(sorted([d1[0], d2[0]]))
        mapping = {"VP": "V-P", "PV": "V-P", "VK": "V-K", "KV": "V-K", "PK": "P-K", "KP": "P-K"}
        return f"Dual ({mapping.get(pair, d1[0] + '-' + d2[0])})"

    return f"Dominant {d1}"

# --- CSV IO (robust) ---
def _flex_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_COLS)
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")

def _safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_new_{int(time.time())}.csv")
        df.to_csv(alt, index=False)
        return alt
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

def load_results() -> pd.DataFrame:
    df = _flex_read_csv(RESULTS_CSV)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[REQUIRED_COLS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["vata_score","pitta_score","kapha_score","vata_%","pitta_%","kapha_%","age","attempt_no"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["name"]   = df["name"].astype(str).str.strip().str.title()
    df["gender"] = df["gender"].astype(str).str.strip().str.title()
    for cc in ["country","state","city","consent_analytics"]:
        df[cc] = df[cc].astype(str).replace({"nan": ""})
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["person_key","timestamp"])
    return df

def save_result_row(row: dict):
    current = load_results()
    for c in REQUIRED_COLS:
        row.setdefault(c, None)
    new = pd.DataFrame([row])
    new["timestamp"] = pd.to_datetime(new["timestamp"], errors="coerce")
    current = pd.concat([current, new], ignore_index=True)
    current["timestamp"] = pd.to_datetime(current["timestamp"], errors="coerce")
    current = current.dropna(subset=["timestamp"])
    current = current.sort_values(["person_key","timestamp"])
    def reassign(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp")
        g["attempt_no"] = range(1, len(g) + 1)
        return g
    try:
        current = current.groupby("person_key", group_keys=False, include_groups=False).apply(reassign)
    except TypeError:
        current = current.groupby("person_key", group_keys=False).apply(reassign)
    _safe_write_csv(current, RESULTS_CSV)

# -------------------------
# DUAL/TRIDOSHA ADVICE
# -------------------------
def dominant_note_and_tips(ptype: str, perc: dict) -> tuple[str, list[str]]:
    base = {
        "Vata": {
            "note": ("<b>Vata</b> qualities (light, dry, cool, mobile) are prominent. "
                     "Support stability, warmth, and moisture with steady routines."),
            "tips": [
                "Prefer warm, cooked, moist foods with good oils; reduce cold, dry, raw foods.",
                "Keep regular mealtimes and a consistent sleep–wake schedule.",
                "Gentle grounding practices (abhyanga with warm oil, restorative yoga).",
                "Protect from cold/dry weather; keep the body warm."
            ]
        },
        "Pitta": {
            "note": ("<b>Pitta</b> qualities (hot, sharp, intense) are prominent. "
                     "Balance heat and intensity with cooling foods and calm routines."),
            "tips": [
                "Favor cooling tastes (sweet, bitter, astringent); limit very spicy/sour/salty.",
                "Eat on time; make lunch the main meal; hydrate well.",
                "Calming practices (prāṇāyāma, time in nature); avoid over-exertion.",
                "Reduce late-night screens and overheating midday sun."
            ]
        },
        "Kapha": {
            "note": ("<b>Kapha</b> qualities (heavy, cool, stable) are prominent. "
                     "Lightness, warmth, and stimulation help prevent sluggishness."),
            "tips": [
                "Favor light, warm, mildly spicy foods; reduce heavy, oily, very sweet foods.",
                "Stay active daily (brisk walk, dynamic yoga); avoid daytime naps.",
                "Supper light/early; include variety and stimulation in routine."
            ]
        }
    }

    def merge(*doshas):
        notes = " & ".join(base[d]["note"] for d in doshas)
        tips = []
        for d in doshas:
            tips += base[d]["tips"][:2]
        seen, out = set(), []
        for t in tips:
            if t not in seen:
                seen.add(t); out.append(t)
        return notes, out

    p = ptype.lower()
    if p.startswith("tridoshic"):
        note = ("Your Prakriti appears <b>Tridoshic</b>: Vata, Pitta and Kapha are relatively balanced. "
                "Maintain balance with moderate routines, seasonal adjustments, and avoiding extremes.")
        tips = [
            "Eat fresh, mostly cooked food; vary by season (warmer/oilier in cold months, lighter/cooling in hot).",
            "Keep regular meals and sleep; include daily movement without over- or under-doing it.",
            "Notice early signs of imbalance (dryness/overthinking, heat/irritability, heaviness/slowness) and adjust diet/lifestyle accordingly."
        ]
        return note, tips

    if p.startswith("dual"):
        pair = ptype.split("(")[-1].strip(")").replace(" ", "")
        d1i, d2i = pair.split("-")
        m = {"V": "Vata", "P": "Pitta", "K": "Kapha"}
        d1, d2 = m.get(d1i, d1i), m.get(d2i, d2i)
        note, tips = merge(d1, d2)
        note = f"Your Prakriti appears <b>Dual</b> — {d1} &amp; {d2}. {note}"
        pair_key = "".join(sorted([d1[0], d2[0]]))
        extra = {
            "PV": "Choose warm but not overly heating foods; keep routines steady and cooling when needed.",
            "PK": "Keep meals light/warm and mildly spicy; avoid very oily/heavy foods and excessive heat.",
            "KV": "Go for warm, light meals with good oils and gentle spices; avoid cold/dry and very heavy foods."
        }
        if extra.get(pair_key):
            tips.insert(0, extra[pair_key])
        return note, tips

    if "vata" in p:
        return (
            "Your Prakriti appears <b>Vata-dominant</b> — creative, quick, and adaptable. "
            "Balance dryness, cold and variability with warm, moist foods and steady routines.",
            base["Vata"]["tips"],
        )
    if "pitta" in p:
        return (
            "Your Prakriti appears <b>Pitta-dominant</b> — energetic, focused, and driven. "
            "Balance excess heat with cooling foods and calm routines.",
            base["Pitta"]["tips"],
        )
    return (
        "Your Prakriti appears <b>Kapha-dominant</b> — steady, calm, and enduring. "
        "Balance heaviness with warmth, lightness, and activity.",
        base["Kapha"]["tips"],
    )

# -------------------------
# PDF
# -------------------------
def build_pdf_report(row: dict, perc: dict, chart_png: Path) -> bytes:
    if not REPORTLAB_OK:
        raise RuntimeError("ReportLab not installed (pip install reportlab)")

    buff = io.BytesIO()
    doc = SimpleDocTemplate(
        buff, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=3.6*cm, bottomMargin=1.6*cm  # fixed header overlap
    )
    styles = getSampleStyleSheet()
    flow = []

    def header(c, d):
        w, h = A4
        c.saveState()
        if LOGO_PATH.exists():
            try:
                c.drawImage(str(LOGO_PATH), w - 4.5*cm, h - 3.5*cm,
                            width=3.5*cm, height=3.5*cm, preserveAspectRatio=True, mask="auto")
            except Exception:
                pass
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2*cm, h - 2.0*cm, BRAND_TITLE)
        c.setFont("Helvetica", 9)
        c.drawString(2*cm, h - 2.6*cm, "Prakriti Assessment Report (Educational)")
        c.restoreState()

    def footer(c, d):
        w, h = A4
        c.saveState()
        c.setFont("Helvetica", 9)
        c.drawCentredString(w/2, 1.2*cm, FOOTER_TEXT)
        c.restoreState()

    def page_decor(c, d): header(c, d); footer(c, d)

    flow.append(Spacer(1, 0.6*cm))

    loc = ", ".join([x for x in [row.get("city",""), row.get("state",""), row.get("country","")] if x]) or "—"
    meta = [
        ["Name", row.get("name","")],
        ["Age", str(row.get("age",""))],
        ["Gender", row.get("gender","")],
        ["Location", loc],
        ["Date/Time", row.get("timestamp","")],
        ["Attempt #", str(row.get("attempt_no",""))],
        ["Type (heuristic)", row.get("type","")],
    ]
    t = Table(meta, hAlign="LEFT", colWidths=[4.0*cm, 10.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(1,0), colors.whitesmoke),
        ("BOX",(0,0),(-1,-1), 0.25, colors.grey),
        ("INNERGRID",(0,0),(-1,-1), 0.25, colors.grey),
        ("FONT",(0,0),(-1,-1),"Helvetica", 10),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.whitesmoke]),
    ]))
    flow += [t, Spacer(1, 0.5*cm)]

    scores_tbl = [
        ["Dosha", "Weighted Score", "Percent"],
        ["Vata",  f"{row['vata_score']}",  f"{perc['Vata']} %"],
        ["Pitta", f"{row['pitta_score']}", f"{perc['Pitta']} %"],
        ["Kapha", f"{row['kapha_score']}", f"{perc['Kapha']} %"],
    ]
    stbl = Table(scores_tbl, hAlign="LEFT", colWidths=[3*cm, 5*cm, 5*cm])
    stbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
        ("BOX",(0,0),(-1,-1), 0.25, colors.grey),
        ("INNERGRID",(0,0),(-1,-1), 0.25, colors.grey),
        ("FONT",(0,0),(-1,-1),"Helvetica", 10),
    ]))
    flow += [stbl, Spacer(1, 0.6*cm)]

    if chart_png and chart_png.exists():
        flow.append(RLImage(str(chart_png), width=12*cm, height=7*cm))
        flow.append(Spacer(1, 0.6*cm))

    note, tips = dominant_note_and_tips(row.get("type",""), perc)

    flow.append(Paragraph("<b>Short interpretation</b>", styles["Heading3"]))
    flow.append(Paragraph(note, styles["BodyText"]))
    flow.append(Spacer(1, 0.2*cm))
    flow.append(Paragraph("<b>General recommendations</b>", styles["Heading3"]))
    for tip in tips:
        flow.append(Paragraph(f"• {tip}", styles["BodyText"]))
    flow.append(Spacer(1, 0.6*cm))

    flow.append(Paragraph("<b>Disclaimer</b>", styles["Heading3"]))
    flow.append(Paragraph(DISCLAIMER, styles["BodyText"]))

    doc.build(flow, onFirstPage=page_decor, onLaterPages=page_decor)
    return buff.getvalue()

# -------------------------
# LOGGING, FEEDBACK, SESSION HELPERS
# -------------------------
def _append_row_csv(row: dict, path: Path, cols: list[str]):
    try:
        if path.exists():
            df = pd.read_csv(path, on_bad_lines="skip")
        else:
            df = pd.DataFrame(columns=cols)
        for c in cols:
            row.setdefault(c, None)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        _safe_write_csv(df[cols], path)
    except Exception as e:
        st.sidebar.warning(f"Write to {path.name} skipped: {e}")

def log_usage(event: str, payload: dict | None = None):
    p = payload or {}
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "name": p.get("name"),
        "gender": p.get("gender"),
        "age": p.get("age"),
        "type": p.get("type"),
        "country": p.get("country"),
        "state": p.get("state"),
        "city": p.get("city"),
    }
    _append_row_csv(row, USAGE_LOG_CSV,
        ["timestamp","event","name","gender","age","type","country","state","city"]
    )

def _set_last_result(ctx: dict): st.session_state["last_result_ctx"] = ctx
def _get_last_result() -> dict | None: return st.session_state.get("last_result_ctx")

# Session id & first log
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess-{int(time.time()*1000)}"
log_usage("app_open", {})

# -------------------------
# APP UI
# -------------------------
st.title("🪷 Prakriti Analyzer — By Kakunje Ayurveda")
mode = st.sidebar.radio("Mode", ["Assessment", "Admin"], horizontal=True)

# ===== ASSESSMENT =====
if mode == "Assessment":
    with st.form("prakriti_form"):
        st.subheader("👤 Basic details")
        c = st.columns(3)
        name   = c[0].text_input("Name")
        age    = c[1].number_input("Age", min_value=5, max_value=120, value=30)
        gender = c[2].selectbox("Gender", ["Male", "Female", "Other"])

        st.markdown("---")
        st.subheader("📍 Optional: Location (helps us understand usage)")
        c2 = st.columns(3)
        country = c2[0].text_input("Country (optional)")
        state   = c2[1].text_input("State/Region (optional)")
        city    = c2[2].text_input("City (optional)")
        consent = st.checkbox(
            "Share anonymous usage analytics (location & basic stats)",
            value=False,
            help="If checked, we log anonymous usage events with the location you typed above."
        )

        st.markdown("---")
        st.subheader("📝 For each question, pick tendency (V/P/K) and intensity")

        scores = {"Vata": 0.0, "Pitta": 0.0, "Kapha": 0.0}
        unanswered = 0

        for idx, (q, opts, cat, q_w) in enumerate(QUESTIONS, start=1):
            st.markdown(f"**{idx}. {q}**")
            cols = st.columns([3, 2])
            choice = cols[0].radio(
                "Tendency",
                options=[f"{opts['Vata']} (Vata)", f"{opts['Pitta']} (Pitta)", f"{opts['Kapha']} (Kapha)"],
                index=None, key=f"q{idx}_dosha", label_visibility="collapsed"
            )
            intensity = cols[1].selectbox(
                "Intensity", options=intensity_labels_for(cat), index=1, key=f"q{idx}_intensity"
            )
            st.caption(f"Question weight: {q_w}×")
            st.divider()

            if choice is None or intensity is None:
                unanswered += 1
            else:
                base = INT_TO_WEIGHT[intensity]
                delta = base * float(q_w)
                if choice.endswith("(Vata)"):  scores["Vata"]  += delta
                elif choice.endswith("(Pitta)"): scores["Pitta"] += delta
                elif choice.endswith("(Kapha)"): scores["Kapha"] += delta

        submitted = st.form_submit_button("Compute Prakriti")

    if submitted:
        if sum(scores.values()) == 0:
            st.warning("Please answer the questions."); st.stop()

        perc = percentify(scores)
        ptype = prakriti_type_from_percent(perc)

        key = person_key(name, age, gender)
        df_existing = load_results()
        attempt_no = int(len(df_existing[df_existing["person_key"] == key]) + 1)

        st.markdown("## 🔎 Result")
        TRI_GAP, DUAL_GAP, DUAL_MIN = get_thresholds()
        st.caption(f"Classification thresholds → Tridoshic ≤{TRI_GAP}%, Dual gap ≤{DUAL_GAP}% with both ≥{DUAL_MIN}%.")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Weighted scores (with question multipliers)**")
            st.json({k: round(v, 2) for k, v in scores.items()})
            st.write("**Percentages**")
            st.json(perc)
            st.success(f"**Type (heuristic):** {ptype}  |  **Attempt:** {attempt_no}")
            if unanswered:
                st.caption(f"⚠️ {unanswered} item(s) unanswered; baseline improves with repeated attempts.")
        with c2:
            fig, ax = plt.subplots()
            ax.bar(["Vata","Pitta","Kapha"], [perc["Vata"], perc["Pitta"], perc["Kapha"]])
            ax.set_ylim(0, 100); ax.set_ylabel("Percent"); ax.set_title("Prakriti Composition")
            chart_path = REPORTS_DIR / f"chart_{int(time.time())}.png"
            fig.savefig(chart_path, bbox_inches="tight", dpi=200)
            st.pyplot(fig, width='stretch')

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "person_key": key, "attempt_no": attempt_no,
            "name": (name or "").title(), "age": age, "gender": (gender or "").title(),
            "country": (country or "").title(), "state": (state or "").title(), "city": (city or "").title(),
            "consent_analytics": bool(consent),
            "vata_score": round(scores["Vata"], 3), "pitta_score": round(scores["Pitta"], 3), "kapha_score": round(scores["Kapha"], 3),
            "vata_%": perc["Vata"], "pitta_%": perc["Pitta"], "kapha_%": perc["Kapha"], "type": ptype
        }
        save_result_row(row)
        st.caption(f"Saved to {RESULTS_CSV}")

        if consent:
            log_usage("assessment_submitted", {
                "name": name, "gender": gender, "age": age, "type": ptype,
                "country": country, "state": state, "city": city
            })

        pdf_bytes, fname = None, None
        if not REPORTLAB_OK:
            st.info("Install ReportLab to enable PDF (pip install reportlab).")
        else:
            try:
                pdf_bytes = build_pdf_report(row, {"Vata": perc["Vata"], "Pitta": perc["Pitta"], "Kapha": perc["Kapha"]}, chart_path)
                fname = f"Prakriti_Report_{row['name']}_{row['timestamp'].replace(':','-')}.pdf"
                st.download_button("📄 Download PDF report (this attempt)", data=pdf_bytes,
                                   file_name=fname, mime="application/pdf", key="dl_current_pdf")
                if consent:
                    log_usage("pdf_downloaded", {
                        "name": name, "gender": gender, "age": age, "type": ptype,
                        "country": country, "state": state, "city": city
                    })
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

        _set_last_result({"row": row, "perc": perc, "ptype": ptype,
                          "chart_path": str(chart_path), "pdf_bytes": pdf_bytes, "pdf_name": fname})

    _last = _get_last_result()
    if _last:
        st.markdown("---")
        st.subheader("📄 Report & Feedback")
        if REPORTLAB_OK and _last.get("pdf_bytes"):
            st.download_button(
                "📄 Download PDF report (last attempt)",
                data=_last["pdf_bytes"], file_name=_last["pdf_name"],
                mime="application/pdf", key="dl_last_pdf",
            )
        elif not REPORTLAB_OK:
            st.info("Install ReportLab to enable PDF (pip install reportlab).")

        with st.expander("💬 Send feedback (for testing)"):
            fb_name = st.text_input("Your name (optional)", value=_last["row"].get("name",""))
            fb_type = st.selectbox("Feedback type", ["Suggestion","Bug","Question","Other"], index=0, key="fb_type_last")
            fb_text = st.text_area("Your feedback", key="fb_text_last")
            if st.button("Submit feedback", key="fb_submit_last"):
                fb_row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "session_id": st.session_state.session_id,
                    "name": (fb_name or "").title(),
                    "feedback_type": fb_type, "feedback": fb_text,
                    "result_type": _last["ptype"],
                    "vata_%": _last["perc"]["Vata"], "pitta_%": _last["perc"]["Pitta"], "kapha_%": _last["perc"]["Kapha"],
                    "country": _last["row"].get("country",""), "state": _last["row"].get("state",""), "city": _last["row"].get("city",""),
                }
                _append_row_csv(
                    fb_row, FEEDBACK_CSV,
                    ["timestamp","session_id","name","feedback_type","feedback","result_type",
                     "vata_%","pitta_%","kapha_%","country","state","city"]
                )
                st.success("Thank you! Feedback recorded.")

# ===== ADMIN =====
if mode == "Admin":
    pin = st.sidebar.text_input("Admin PIN", type="password")
    if pin != ADMIN_PIN:
        st.warning("Enter correct PIN to access Admin."); st.stop()

    # ---- Live thresholds (persisted) ----
    st.sidebar.markdown("### Classification thresholds")
    cfg = st.session_state.config
    tri_gap  = st.sidebar.slider("TRI_GAP  (max spread between all three for Tridoshic)",
                                 min_value=1.0, max_value=15.0, value=float(cfg["TRI_GAP"]), step=0.5)
    dual_gap = st.sidebar.slider("DUAL_GAP (max gap between top two for Dual)",
                                 min_value=1.0, max_value=15.0, value=float(cfg["DUAL_GAP"]), step=0.5)
    dual_min = st.sidebar.slider("DUAL_MIN (minimum % both top doshas must meet)",
                                 min_value=10.0, max_value=60.0, value=float(cfg["DUAL_MIN"]), step=1.0)

    if (tri_gap, dual_gap, dual_min) != (cfg["TRI_GAP"], cfg["DUAL_GAP"], cfg["DUAL_MIN"]):
        st.session_state.config = {"TRI_GAP": tri_gap, "DUAL_GAP": dual_gap, "DUAL_MIN": dual_min}
        save_config(st.session_state.config)
        st.sidebar.success("Thresholds saved. New assessments will use the updated values.")

    st.info(f"Current thresholds → TRI_GAP: {st.session_state.config['TRI_GAP']}, "
            f"DUAL_GAP: {st.session_state.config['DUAL_GAP']}, "
            f"DUAL_MIN: {st.session_state.config['DUAL_MIN']}")

    # ---- Results table & charts ----
    df = load_results()
    st.header("🧭 Admin — Results Browser")

    name_filter = st.sidebar.text_input("Search name contains")
    genders = df["gender"].dropna().unique().tolist()
    gsel = st.sidebar.multiselect("Gender", options=genders, default=genders if genders else [])
    types = df["type"].dropna().unique().tolist()
    tsel = st.sidebar.multiselect("Prakriti type", options=types, default=types if types else [])
    countries = sorted([x for x in df["country"].dropna().unique().tolist() if x])
    csel = st.sidebar.multiselect("Country", options=countries, default=countries if countries else [])

    q = df.copy()
    if name_filter: q = q[q["name"].fillna("").str.contains(name_filter, case=False)]
    if gsel:        q = q[q["gender"].isin(gsel)]
    if tsel:        q = q[q["type"].isin(tsel)]
    if csel:        q = q[q["country"].isin(csel)]

    st.subheader("Filtered Results")
    st.dataframe(q.sort_values(["timestamp"], ascending=False), width='stretch')

    col = st.columns(5)
    with col[0]: st.metric("Records", len(q))
    with col[1]: st.metric("Avg Vata %",  f"{q['vata_%'].mean():.1f}" if len(q) else "—")
    with col[2]: st.metric("Avg Pitta %", f"{q['pitta_%'].mean():.1f}" if len(q) else "—")
    with col[3]: st.metric("Avg Kapha %", f"{q['kapha_%'].mean():.1f}" if len(q) else "—")
    with col[4]: st.metric("Countries", q['country'].nunique() if len(q) else 0)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distribution by Type**")
        counts = q["type"].value_counts().sort_index()
        fig, ax = plt.subplots()
        if counts.empty:
            ax.text(0.5, 0.5, "No data to display (check filters)", ha="center", va="center"); ax.axis("off")
        else:
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Count"); ax.set_xlabel("Type"); ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig, width='stretch')

    with c2:
        st.markdown("**Average Dosha %**")
        fig2, ax2 = plt.subplots()
        if q.empty:
            ax2.text(0.5, 0.5, "No data to display (check filters)", ha="center", va="center"); ax2.axis("off")
        else:
            ax2.bar(["Vata","Pitta","Kapha"], [q["vata_%"].mean(), q["pitta_%"].mean(), q["kapha_%"].mean()])
            ax2.set_ylim(0, 100); ax2.set_ylabel("Percent"); ax2.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig2, width='stretch')

    st.markdown("---")
    st.subheader("📄 Generate PDF for a past attempt")
    if q.empty:
        st.info("No records to export. Run an assessment first.")
    else:
        q_sorted = q.sort_values("timestamp", ascending=False).reset_index(drop=True)
        labels = [
            f"{i+1}. {r['name']} | {r['gender']} | Age {int(r['age']) if pd.notna(r['age']) else '—'} | "
            f"{r['type']} | {pd.to_datetime(r['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} "
            f"(Attempt {int(r['attempt_no']) if pd.notna(r['attempt_no']) else '—'})"
            for i, r in q_sorted.iterrows()
        ]
        sel = st.selectbox("Select a record", options=list(range(len(labels))), format_func=lambda i: labels[i])

        if st.button("Generate PDF for selected record"):
            rec = q_sorted.iloc[int(sel)].to_dict()
            perc_sel = {"Vata": rec["vata_%"], "Pitta": rec["pitta_%"], "Kapha": rec["kapha_%"]}

            figx, axx = plt.subplots()
            axx.bar(["Vata","Pitta","Kapha"], [perc_sel["Vata"], perc_sel["Pitta"], perc_sel["Kapha"]])
            axx.set_ylim(0, 100); axx.set_ylabel("Percent"); axx.set_title("Prakriti Composition")
            chart_sel = REPORTS_DIR / f"chart_{int(time.time())}_sel.png"
            figx.savefig(chart_sel, bbox_inches="tight", dpi=200)
            st.pyplot(figx, width='stretch')

            if not REPORTLAB_OK:
                st.error("Install ReportLab to enable PDF (pip install reportlab).")
            else:
                try:
                    pdf_bytes = build_pdf_report(rec, perc_sel, chart_sel)
                    fname = f"Prakriti_Report_{rec['name']}_{str(rec['timestamp']).replace(':','-')}.pdf"
                    st.download_button("📄 Download PDF (selected record)", data=pdf_bytes,
                                       file_name=fname, mime="application/pdf")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

    st.markdown("---")
    st.subheader("📝 Feedback (testing)")
    try:
        if FEEDBACK_CSV.exists():
            dff = pd.read_csv(FEEDBACK_CSV, on_bad_lines="skip")
            st.dataframe(dff.sort_values("timestamp", ascending=False), width='stretch')
            st.download_button("⬇️ Download feedback CSV", data=dff.to_csv(index=False).encode("utf-8"),
                               file_name="feedback_export.csv", mime="text/csv")
        else:
            st.info("No feedback yet.")
    except Exception as e:
        st.warning(f"Could not load feedback: {e}")

    st.subheader("📊 Usage Analytics (anonymous, consented only)")
    try:
        if USAGE_LOG_CSV.exists():
            ul = pd.read_csv(USAGE_LOG_CSV, on_bad_lines="skip")
            ul["timestamp"] = pd.to_datetime(ul["timestamp"], errors="coerce")
            ul["day"] = ul["timestamp"].dt.date
            per_day = ul.groupby("day")["event"].count()

            top_countries = ul["country"].fillna("").replace({"nan": ""})
            top_counts = top_countries[top_countries != ""].value_counts().head(10)

            mc1, mc2, mc3 = st.columns(3)
            with mc1: st.metric("Events logged", len(ul))
            with mc2: st.metric("Unique days", per_day.index.nunique())
            with mc3: st.metric("Countries seen", top_countries[top_countries != ""].nunique())

            figd, axd = plt.subplots()
            if per_day.empty:
                axd.text(0.5, 0.5, "No events yet", ha="center", va="center"); axd.axis("off")
            else:
                per_day.plot(kind="bar", ax=axd)
                axd.set_ylabel("Events"); axd.set_xlabel("Day"); axd.set_title("Events per day")
                axd.grid(True, axis="y", alpha=0.3)
            st.pyplot(figd, width='stretch')

            figc, axc = plt.subplots()
            if top_counts.empty:
                axc.text(0.5, 0.5, "No country data yet", ha="center", va="center"); axc.axis("off")
            else:
                top_counts.plot(kind="bar", ax=axc)
                axc.set_ylabel("Events"); axc.set_xlabel("Country"); axc.setTitle = "Top countries"
                axc.grid(True, axis="y", alpha=0.3)
            st.pyplot(figc, width='stretch')

            st.download_button("⬇️ Download usage log CSV",
                               data=ul.to_csv(index=False).encode("utf-8"),
                               file_name="usage_log_export.csv", mime="text/csv")
        else:
            st.info("No events logged yet.")
    except Exception as e:
        st.warning(f"Could not load usage analytics: {e}")
