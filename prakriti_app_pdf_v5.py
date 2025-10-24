# prakriti_app_pdf_v5.py — Prakriti Analyzer (Weighted, 15 Qs, PDF, Final)
# Brand: Kakunje Ayurveda Clinic & Research Centre
# -----------------------------------------------------------------------
# Features
#  • 15 clearer questions (less ambiguity), per-question weights
#  • Adaptive intensity labels (physical / variable / psychological)
#  • Safe CSV writes (atomic) to data/prakriti_results.csv
#  • Timestamp dtype fix + pandas groupby (new/old versions)
#  • Admin dashboard: filters, charts, and "Generate PDF for past attempt"
#  • PDF with logo header, clinic footer, short note, recommendations, disclaimer
#  • Streamlit ≥2025 compliant (st.pyplot(..., width='stretch'))
# -----------------------------------------------------------------------

import os, time, io, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional (but recommended) to keep console clean
warnings.filterwarnings("ignore")

# PDF dependency
REPORTLAB_OK = True
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image as RLImage,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors
except Exception:
    REPORTLAB_OK = False

# -------------------------
# CONFIG
# -------------------------
BRAND_TITLE = "Kakunje Ayurveda Clinic & Research Centre"
LOGO_PATH = Path("logo.png")  # put your logo file in same folder as this .py
DATA_DIR = Path("data")
REPORTS_DIR = DATA_DIR / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = DATA_DIR / "prakriti_results.csv"
ADMIN_PIN = "1234"  # change if needed

# --- Footer text for PDF reports (your final details) ---
FOOTER_TEXT = "Kakunje Ayurveda Clinic & Research Centre · Moodubidri, Karnataka, India · +91-9483697676 · kakunje.com · prasanna@kakunje.com"

REQUIRED_COLS = [
    "timestamp",
    "person_key",
    "attempt_no",
    "name",
    "age",
    "gender",
    "vata_score",
    "pitta_score",
    "kapha_score",
    "vata_%",
    "pitta_%",
    "kapha_%",
    "type",
]

# intensity → base weights
INT_TO_WEIGHT = {
    "Rarely": 0,
    "Sometimes": 1,
    "Often": 2,  # variable
    "Low": 0,
    "Moderate": 1,
    "High": 2,  # psychological
    "Mildly": 0,
    "Moderately": 1,
    "Strongly": 2,  # physical
}

# -------------------------
# 15 QUESTIONS — clear, non-overlapping phrasing
# Each: (question, {Vata,Pitta,Kapha}, category, weight)
# -------------------------
QUESTIONS = [
    (
        "Natural body build",
        {
            "Vata": "Thin/lean; hard to gain weight",
            "Pitta": "Medium/muscular; moderate weight",
            "Kapha": "Broad/heavier; gains easily",
        },
        "physical",
        1.0,
    ),
    (
        "Skin baseline (without products)",
        {
            "Vata": "Dry/rough/cool",
            "Pitta": "Warm/sensitive; redness possible",
            "Kapha": "Smooth/thick/oily",
        },
        "physical",
        1.0,
    ),
    (
        "Hair & scalp tendency",
        {
            "Vata": "Dry hair/scalp; frizz",
            "Pitta": "Fine hair; early greying common",
            "Kapha": "Thick/lustrous; oily scalp",
        },
        "physical",
        0.8,
    ),
    (
        "Appetite pattern (day to day)",
        {
            "Vata": "Irregular; sometimes low/sometimes high",
            "Pitta": "Strong/sharp; hungry on time",
            "Kapha": "Slow/steady; can skip meals",
        },
        "variable",
        1.6,
    ),
    (
        "Digestion / bowel pattern",
        {
            "Vata": "Variable; gas/bloating common",
            "Pitta": "Fast; tendency to loose/acidic",
            "Kapha": "Slower; well-formed stools",
        },
        "variable",
        1.3,
    ),
    (
        "Thirst & sweating",
        {
            "Vata": "Low thirst; minimal sweat",
            "Pitta": "High thirst; profuse sweat",
            "Kapha": "Moderate thirst; mild sweat",
        },
        "variable",
        1.0,
    ),
    (
        "Thermal comfort",
        {
            "Vata": "Prefers warmth; dislikes cold",
            "Pitta": "Feels hot easily; prefers cool",
            "Kapha": "Comfortable with cool; dislikes damp cold",
        },
        "physical",
        0.9,
    ),
    (
        "Sleep quality",
        {
            "Vata": "Light; wakes easily; mind active",
            "Pitta": "Moderate; heat/dreams may disturb",
            "Kapha": "Deep/heavy; longer duration",
        },
        "variable",
        1.3,
    ),
    (
        "Mental focus (typical)",
        {
            "Vata": "Quick ideas; distractible",
            "Pitta": "Sharp focus; task-driven",
            "Kapha": "Steady; slower but consistent",
        },
        "psychological",
        1.5,
    ),
    (
        "Emotional response under stress",
        {
            "Vata": "Worry/anxiety; overthinking",
            "Pitta": "Irritability/anger; impatience",
            "Kapha": "Withdrawal/attachment; lethargy",
        },
        "psychological",
        1.4,
    ),
    (
        "Speech & pace",
        {
            "Vata": "Fast/variable; animated",
            "Pitta": "Clear/precise; firm",
            "Kapha": "Slow/steady; calm",
        },
        "variable",
        0.8,
    ),
    (
        "Joints & movement",
        {
            "Vata": "Cracking/mobility; dryness",
            "Pitta": "Warm; occasional tenderness",
            "Kapha": "Stable/cushioned; less cracking",
        },
        "physical",
        0.9,
    ),
    (
        "Food preferences",
        {
            "Vata": "Warm, oily, grounding foods",
            "Pitta": "Cooling/light foods; dislikes very spicy",
            "Kapha": "Light/warm/spicy foods; dislikes heavy",
        },
        "variable",
        1.0,
    ),
    (
        "Weather preference",
        {
            "Vata": "Warm/dry climate",
            "Pitta": "Cool/mild climate",
            "Kapha": "Cold/dry climate with sunshine",
        },
        "variable",
        0.7,
    ),
    (
        "Typical energy pattern",
        {
            "Vata": "Variable energy; bursts and dips",
            "Pitta": "Strong/consistent until late crash",
            "Kapha": "Steady but slow; builds with activity",
        },
        "psychological",
        1.2,
    ),
]


# -------------------------
# HELPERS
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


def prakriti_type_from_percent(perc: dict) -> str:
    items = sorted(perc.items(), key=lambda x: x[1], reverse=True)
    top, mid, low = items[0][1], items[1][1], items[2][1]
    CLOSE = 5.0
    if abs(top - mid) <= CLOSE and abs(mid - low) <= CLOSE:
        return "Tridoshic (V-P-K)"
    if abs(top - mid) <= CLOSE:
        return f"Dual ({items[0][0][0]}-{items[1][0][0]})"
    return f"Dominant {items[0][0]}"


# Robust CSV IO
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
        os.replace(tmp, path)  # atomic on Windows
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
    for c in [
        "vata_score",
        "pitta_score",
        "kapha_score",
        "vata_%",
        "pitta_%",
        "kapha_%",
        "age",
        "attempt_no",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["name"] = df["name"].astype(str).str.strip().str.title()
    df["gender"] = df["gender"].astype(str).str.strip().str.title()
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["person_key", "timestamp"])
    return df


def save_result_row(row: dict):
    # Load current
    current = load_results()
    # Ensure row has all columns
    for c in REQUIRED_COLS:
        row.setdefault(c, None)
    # New DF and coerce timestamp
    new = pd.DataFrame([row])
    new["timestamp"] = pd.to_datetime(new["timestamp"], errors="coerce")
    # Append and coerce combined timestamp
    current = pd.concat([current, new], ignore_index=True)
    current["timestamp"] = pd.to_datetime(current["timestamp"], errors="coerce")
    current = current.dropna(subset=["timestamp"])
    # Sort and reassign attempts
    current = current.sort_values(["person_key", "timestamp"])

    def reassign(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp")
        g["attempt_no"] = range(1, len(g) + 1)
        return g

    # Compatible with old/new pandas
    try:
        current = current.groupby(
            "person_key", group_keys=False, include_groups=False
        ).apply(reassign)
    except TypeError:
        current = current.groupby("person_key", group_keys=False).apply(reassign)

    _safe_write_csv(current, RESULTS_CSV)


# -------------------------
# PDF GENERATION
# -------------------------
def dominant_note_and_tips(perc: dict) -> tuple[str, list[str]]:
    dom = max(perc, key=perc.get)
    if dom == "Pitta":
        note = (
            "Your Prakriti appears to be <b>Pitta-dominant</b> — energetic, focused, "
            "and driven. Balance excessive heat or irritability with cooling foods, "
            "regular meals, adequate hydration, and calm routines."
        )
        tips = [
            "Prefer cooling foods (sweet, bitter, astringent); reduce very spicy/sour/salty.",
            "Avoid skipping meals; take lunch as main meal.",
            "Practice calming activities (prāṇāyāma, gentle yoga, time in nature).",
            "Keep work–rest balance; avoid overexertion and late-night screens.",
        ]
    elif dom == "Vata":
        note = (
            "Your Prakriti appears to be <b>Vata-dominant</b> — creative, quick, and adaptable. "
            "Balance dryness, cold and variability with warm, moist foods and steady routines."
        )
        tips = [
            "Prefer warm, cooked, moist foods with good oils; reduce raw, cold, dry foods.",
            "Keep regular mealtimes and sleep schedule.",
            "Gentle, grounding practices (abhyanga oil massage, restorative yoga).",
            "Protect from cold/dry weather; keep the body warm.",
        ]
    else:  # Kapha
        note = (
            "Your Prakriti appears to be <b>Kapha-dominant</b> — steady, calm, and enduring. "
            "Balance heaviness and sluggishness with warmth, lightness, and stimulation."
        )
        tips = [
            "Favor light, warm, and mildly spicy foods; reduce heavy, oily, and sweet foods.",
            "Stay active daily; brisk walking, dynamic yoga.",
            "Avoid daytime sleep; take lighter dinners.",
            "Keep variety and stimulation in routines to avoid lethargy.",
        ]
    return note, tips


DISCLAIMER = (
    "This assessment is educational and not a medical diagnosis. "
    "Please consult a qualified Ayurveda physician for personalized advice and treatment."
)


def build_pdf_report(row: dict, perc: dict, chart_png: Path) -> bytes:
    """
    Build a single-page PDF with header (logo+title), footer (clinic info),
    scores table, chart image, short interpretation, tips, and disclaimer.
    """
    if not REPORTLAB_OK:
        raise RuntimeError(
            "ReportLab not installed. Install it with: pip install reportlab"
        )

    buff = io.BytesIO()
    doc = SimpleDocTemplate(
        buff,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    flow = []

    # Header
    def header(canvas, doc):
        canvas.saveState()
        w, h = A4
        if LOGO_PATH.exists():
            try:
                canvas.drawImage(
                    str(LOGO_PATH),
                    w - 4.5 * cm,
                    h - 3.5 * cm,
                    width=3.5 * cm,
                    height=3.5 * cm,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception:
                pass
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(2 * cm, h - 2.0 * cm, BRAND_TITLE)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(
            2 * cm, h - 2.6 * cm, "Prakriti Assessment Report (Educational)"
        )
        canvas.restoreState()

    # Footer
    def footer(canvas, doc):
        w, h = A4
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(w / 2, 1.2 * cm, FOOTER_TEXT)
        canvas.restoreState()

    def page_decor(canvas, doc):
        header(canvas, doc)
        footer(canvas, doc)

    # Body
    flow.append(Spacer(1, 2.2 * cm))

    meta = [
        ["Name", row.get("name", "")],
        ["Age", str(row.get("age", ""))],
        ["Gender", row.get("gender", "")],
        ["Date/Time", row.get("timestamp", "")],
        ["Attempt #", str(row.get("attempt_no", ""))],
        ["Type (heuristic)", row.get("type", "")],
    ]
    t = Table(meta, hAlign="LEFT", colWidths=[4.0 * cm, 10.5 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (1, 0), colors.whitesmoke),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
            ]
        )
    )
    flow += [t, Spacer(1, 0.5 * cm)]

    scores_tbl = [
        ["Dosha", "Weighted Score", "Percent"],
        ["Vata", f"{row['vata_score']}", f"{perc['Vata']} %"],
        ["Pitta", f"{row['pitta_score']}", f"{perc['Pitta']} %"],
        ["Kapha", f"{row['kapha_score']}", f"{perc['Kapha']} %"],
    ]
    stbl = Table(scores_tbl, hAlign="LEFT", colWidths=[3 * cm, 5 * cm, 5 * cm])
    stbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
            ]
        )
    )
    flow += [stbl, Spacer(1, 0.6 * cm)]

    if chart_png and chart_png.exists():
        flow.append(RLImage(str(chart_png), width=12 * cm, height=7 * cm))
        flow.append(Spacer(1, 0.6 * cm))

    note, tips = dominant_note_and_tips(perc)
    flow.append(Paragraph("<b>Short interpretation</b>", styles["Heading3"]))
    flow.append(Paragraph(note, styles["BodyText"]))
    flow.append(Spacer(1, 0.2 * cm))
    flow.append(Paragraph("<b>General recommendations</b>", styles["Heading3"]))
    for tip in tips:
        flow.append(Paragraph(f"• {tip}", styles["BodyText"]))
    flow.append(Spacer(1, 0.6 * cm))

    flow.append(Paragraph("<b>Disclaimer</b>", styles["Heading3"]))
    flow.append(Paragraph(DISCLAIMER, styles["BodyText"]))

    doc.build(flow, onFirstPage=page_decor, onLaterPages=page_decor)
    return buff.getvalue()


# -------------------------
# APP
# -------------------------
st.set_page_config(page_title="Prakriti Analyzer — PDF", layout="wide")
st.title("🪷 Prakriti Analyzer — By Kakunje Ayurveda")

mode = st.sidebar.radio("Mode", ["Assessment", "Admin"], horizontal=True)

# ---------- ASSESSMENT ----------
if mode == "Assessment":
    with st.form("prakriti_form"):
        st.subheader("👤 Basic details")
        c = st.columns(3)
        name = c[0].text_input("Name")
        age = c[1].number_input("Age", min_value=5, max_value=120, value=30)
        gender = c[2].selectbox("Gender", ["Male", "Female", "Other"])

        st.markdown("---")
        st.subheader(
            "📝 For each question, pick **tendency** (V/P/K) and **intensity**"
        )

        scores = {"Vata": 0.0, "Pitta": 0.0, "Kapha": 0.0}
        unanswered = 0

        for idx, (q, opts, cat, q_w) in enumerate(QUESTIONS, start=1):
            st.markdown(f"**{idx}. {q}**")
            cols = st.columns([3, 2])
            choice = cols[0].radio(
                "Tendency",
                options=[
                    f"{opts['Vata']} (Vata)",
                    f"{opts['Pitta']} (Pitta)",
                    f"{opts['Kapha']} (Kapha)",
                ],
                index=None,
                key=f"q{idx}_dosha",
                label_visibility="collapsed",
            )
            intensity = cols[1].selectbox(
                "Intensity",
                options=intensity_labels_for(cat),
                index=1,
                key=f"q{idx}_intensity",
            )
            st.caption(f"Question weight: {q_w}×")
            st.divider()

            if choice is None or intensity is None:
                unanswered += 1
            else:
                base = INT_TO_WEIGHT[intensity]  # 0/1/2
                delta = base * float(q_w)
                if choice.endswith("(Vata)"):
                    scores["Vata"] += delta
                elif choice.endswith("(Pitta)"):
                    scores["Pitta"] += delta
                elif choice.endswith("(Kapha)"):
                    scores["Kapha"] += delta

        submitted = st.form_submit_button("Compute Prakriti")

    if submitted:
        if sum(scores.values()) == 0:
            st.warning("Please answer the questions.")
            st.stop()

        perc = percentify(scores)
        ptype = prakriti_type_from_percent(perc)

        key = person_key(name, age, gender)
        df_existing = load_results()
        attempt_no = int(len(df_existing[df_existing["person_key"] == key]) + 1)

        st.markdown("## 🔎 Result")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Weighted scores (with question multipliers)**")
            st.json({k: round(v, 2) for k, v in scores.items()})
            st.write("**Percentages**")
            st.json(perc)
            st.success(f"**Type (heuristic):** {ptype}  |  **Attempt:** {attempt_no}")
            if unanswered:
                st.caption(
                    f"⚠️ {unanswered} item(s) unanswered; baseline improves with repeated attempts."
                )
        with c2:
            fig, ax = plt.subplots()
            ax.bar(
                ["Vata", "Pitta", "Kapha"], [perc["Vata"], perc["Pitta"], perc["Kapha"]]
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel("Percent")
            ax.set_title("Prakriti Composition")
            chart_path = REPORTS_DIR / f"chart_{int(time.time())}.png"
            fig.savefig(chart_path, bbox_inches="tight", dpi=200)
            st.pyplot(fig, width="stretch")

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "person_key": key,
            "attempt_no": attempt_no,
            "name": (name or "").title(),
            "age": age,
            "gender": (gender or "").title(),
            "vata_score": round(scores["Vata"], 3),
            "pitta_score": round(scores["Pitta"], 3),
            "kapha_score": round(scores["Kapha"], 3),
            "vata_%": perc["Vata"],
            "pitta_%": perc["Pitta"],
            "kapha_%": perc["Kapha"],
            "type": ptype,
        }
        save_result_row(row)
        st.caption(f"Saved to {RESULTS_CSV}")

        # PDF for current attempt
        if not REPORTLAB_OK:
            st.error("To enable PDF, install ReportLab:\n\n`pip install reportlab`")
        else:
            try:
                pdf_bytes = build_pdf_report(
                    row,
                    {
                        "Vata": perc["Vata"],
                        "Pitta": perc["Pitta"],
                        "Kapha": perc["Kapha"],
                    },
                    chart_path,
                )
                file_name = f"Prakriti_Report_{row['name']}_{row['timestamp'].replace(':','-')}.pdf"
                st.download_button(
                    "📄 Download PDF report (this attempt)",
                    data=pdf_bytes,
                    file_name=file_name,
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

# ---------- ADMIN ----------
if mode == "Admin":
    pin = st.sidebar.text_input("Admin PIN", type="password")
    if pin != ADMIN_PIN:
        st.warning("Enter correct PIN to access Admin.")
        st.stop()

    df = load_results()
    st.header("🧭 Admin — Results Browser")

    # Filters
    name_filter = st.sidebar.text_input("Search name contains")
    genders = df["gender"].dropna().unique().tolist()
    gsel = st.sidebar.multiselect(
        "Gender", options=genders, default=genders if genders else []
    )
    types = df["type"].dropna().unique().tolist()
    tsel = st.sidebar.multiselect(
        "Prakriti type", options=types, default=types if types else []
    )

    q = df.copy()
    if name_filter:
        q = q[q["name"].fillna("").str.contains(name_filter, case=False)]
    if gsel:
        q = q[q["gender"].isin(gsel)]
    if tsel:
        q = q[q["type"].isin(tsel)]

    st.subheader("Filtered Results")
    st.dataframe(q.sort_values(["timestamp"], ascending=False), width="stretch")

    col = st.columns(4)
    with col[0]:
        st.metric("Records", len(q))
    with col[1]:
        st.metric("Avg Vata %", f"{q['vata_%'].mean():.1f}" if len(q) else "—")
    with col[2]:
        st.metric("Avg Pitta %", f"{q['pitta_%'].mean():.1f}" if len(q) else "—")
    with col[3]:
        st.metric("Avg Kapha %", f"{q['kapha_%'].mean():.1f}" if len(q) else "—")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distribution by Type**")
        counts = q["type"].value_counts().sort_index()
        fig, ax = plt.subplots()
        if counts.empty:
            ax.text(
                0.5, 0.5, "No data to display (check filters)", ha="center", va="center"
            )
            ax.axis("off")
        else:
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel("Type")
            ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig, width="stretch")

    with c2:
        st.markdown("**Average Dosha %**")
        fig2, ax2 = plt.subplots()
        if q.empty:
            ax2.text(
                0.5, 0.5, "No data to display (check filters)", ha="center", va="center"
            )
            ax2.axis("off")
        else:
            ax2.bar(
                ["Vata", "Pitta", "Kapha"],
                [q["vata_%"].mean(), q["pitta_%"].mean(), q["kapha_%"].mean()],
            )
            ax2.set_ylim(0, 100)
            ax2.set_ylabel("Percent")
            ax2.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig2, width="stretch")

    st.markdown("---")
    st.subheader("📄 Generate PDF for a past attempt")

    if q.empty:
        st.info("No records to export. Run an assessment first.")
    else:
        # Build a simple selector with readable labels
        q_sorted = q.sort_values("timestamp", ascending=False).reset_index(drop=True)
        labels = [
            f"{i+1}. {r['name']} | {r['gender']} | Age {int(r['age']) if pd.notna(r['age']) else '—'} | "
            f"{r['type']} | {pd.to_datetime(r['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} (Attempt {int(r['attempt_no']) if pd.notna(r['attempt_no']) else '—'})"
            for i, r in q_sorted.iterrows()
        ]
        sel = st.selectbox(
            "Select a record",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
        )

        if st.button("Generate PDF for selected record"):
            rec = q_sorted.iloc[int(sel)].to_dict()
            perc_sel = {
                "Vata": rec["vata_%"],
                "Pitta": rec["pitta_%"],
                "Kapha": rec["kapha_%"],
            }

            # Build a temporary chart for this record
            figx, axx = plt.subplots()
            axx.bar(
                ["Vata", "Pitta", "Kapha"],
                [perc_sel["Vata"], perc_sel["Pitta"], perc_sel["Kapha"]],
            )
            axx.set_ylim(0, 100)
            axx.set_ylabel("Percent")
            axx.set_title("Prakriti Composition")
            chart_sel = REPORTS_DIR / f"chart_{int(time.time())}_sel.png"
            figx.savefig(chart_sel, bbox_inches="tight", dpi=200)
            st.pyplot(figx, width="stretch")

            if not REPORTLAB_OK:
                st.error("To enable PDF, install ReportLab:\n\n`pip install reportlab`")
            else:
                try:
                    pdf_bytes = build_pdf_report(rec, perc_sel, chart_sel)
                    fname = f"Prakriti_Report_{rec['name']}_{str(rec['timestamp']).replace(':','-')}.pdf"
                    st.download_button(
                        "📄 Download PDF (selected record)",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
