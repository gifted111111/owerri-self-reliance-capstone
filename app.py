import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Owerri Student Self-Reliance Capstone",
    page_icon="🎓",
    layout="wide",
)

MODEL_PATH = "model.pkl"
METRICS_PATH = "metrics.json"
DATA_PATH = "student_constraints.csv"

PHASE_INFO = {
    0: {
        "name": "Phase 1: Environment & Energy Optimization",
        "focus": "Overcoming erratic electricity, unstable network access, and high data costs.",
        "actions": [
            "Batch high-power and high-data tasks (downloads, model runs, CAD exports) during guaranteed power or strong-signal windows.",
            "Keep offline-first copies of lecture notes, documentation, and local scripts so study never stalls during blackouts.",
            "Use lightweight browser/app modes, compress files before syncing, and schedule deep work around peak focus hours.",
        ],
    },
    1: {
        "name": "Phase 2: Virtual & Low-Cost Hardware Labs",
        "focus": "Bridging missing or crowded campus laboratories and expensive equipment.",
        "actions": [
            "Use free cloud notebooks (such as Google Colab), simulators, and virtual sandboxes when physical lab hardware is unavailable.",
            "Build small, modular prototypes with low-cost or readily available components before scaling up.",
            "Validate core logic and simulation behavior first to avoid wasted time and costly mistakes.",
        ],
    },
    2: {
        "name": "Phase 3: Peer-Driven Execution",
        "focus": "Eliminating solo bottlenecks, guesswork, and accountability drop-off.",
        "actions": [
            "Form a small 2–4 person accountability pod with clear weekly deliverables and short check-ins.",
            "Split research, debugging, and documentation tasks, then review each other's work to catch blind spots early.",
            "Set fixed weekly milestones so small consistent wins compound into a completed project.",
        ],
    },
}

DEVICE_LABELS = {
    0: "Smartphone Only",
    1: "Low-Spec Laptop",
    2: "Standard / High-Spec Laptop",
}

LAB_LABELS = {
    0: "None / Unavailable",
    1: "Limited / Crowded",
    2: "Regular Access",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics_and_data():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    df = pd.read_csv(DATA_PATH)
    return metrics, df


def build_refined_prompt(goal, device_label, power_hrs, data_mb, lab_label, peer_score, practice_hrs, phase_name):
    return f"""Act as a practical academic and technical mentor for a student in Owerri facing infrastructure constraints.
Student Profile:
- Target Skill / Course Goal: {goal}
- Primary Device: {device_label}
- Average Daily Electricity: {power_hrs} hrs/day
- Daily Data Budget: {data_mb} MB/day
- Campus Lab Access: {lab_label}
- Peer Study Support Score (1-5): {peer_score}/5
- Current Weekly Practice Time: {practice_hrs} hrs/week
- Diagnosed Primary Framework Phase: {phase_name}

Task:
Create a concise, realistic 7-day self-reliance action plan for "{goal}" that prioritizes {phase_name} while respecting the student's power, data, and hardware limits. Include daily micro-tasks, offline/low-data workarounds, and one measurable weekly output."""


def generate_weekly_plan(goal, device_tier, power_hrs, data_mb, lab_access, peer_score, practice_hrs, phase_id):
    phase = PHASE_INFO[phase_id]

    if device_tier == 0:
        tool_tip = "Use mobile-friendly coding/reading tools, outline logic on paper/notes, and test compact scripts or cloud notebooks when data allows."
    elif device_tier == 1:
        tool_tip = "Keep background apps closed, use lightweight editors/environments, and run heavy tasks in short focused batches."
    else:
        tool_tip = "Use local development environments alongside cloud notebooks to keep a fast workflow both online and offline."

    return [
        ("Day 1 — Baseline & Setup", f"Define a concrete mini-target for **{goal}** and organize offline reference files. {tool_tip}"),
        ("Day 2 — Core Concept & Low-Data Prep", f"Study one core concept of **{goal}** using downloaded/offline materials so you don't burn data while reading."),
        ("Day 3 — Guided Mini-Exercise", f"Complete a small hands-on exercise focused on **{phase['name']}**—{phase['actions'][0].lower()}"),
        ("Day 4 — Bottleneck Workaround", f"Apply **{phase['actions'][1].lower()}** Keep notes on any error, blocker, or resource limit you hit."),
        ("Day 5 — Peer Check-In & Debugging", f"Share your progress or blocker with a classmate/peer group (current peer score: {peer_score}/5) and fix your top issue."),
        ("Day 6 — Prototype / Practice Sprint", f"Dedicate a focused block to building a small working piece for **{goal}** within your {practice_hrs} hrs/week pace."),
        ("Day 7 — Review & Next-Week Prompt", f"Test what you built, summarize lessons learned, and refine your next weekly goal based on actual power ({power_hrs}h) and data ({data_mb}MB) availability."),
    ]


st.title("🎓 Owerri Student Self-Reliance & Study-Pathway Recommender")
st.caption("AI & Machine Learning Capstone Project — Translating the Three-Phase Self-Reliance Framework into an interactive diagnostic and planning tool.")

if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH) and os.path.exists(DATA_PATH)):
    st.error("Missing `model.pkl`, `metrics.json`, or `student_constraints.csv`. Run `generate_data.py` and `train_model.py` first.")
    st.stop()

model = load_model()
metrics, df = load_metrics_and_data()

tab1, tab2 = st.tabs(["🧭 Student Recommender & Action Plan", "📊 Defense Metrics & Model Evaluation"])

with tab1:
    col_input, col_output = st.columns([1, 1.4], gap="large")

    with col_input:
        st.subheader("1. Enter Student Constraints & Goal")
        goal = st.text_input(
            "Target Course / Skill / Project Goal",
            value="Python & Machine Learning Fundamentals",
            help="e.g., Python, Embedded Systems, CAD Modeling, Data Analysis, Exam Prep",
        )
        device_tier = st.selectbox(
            "Primary Device Access",
            options=[0, 1, 2],
            format_func=lambda x: DEVICE_LABELS[x],
            index=1,
        )
        daily_power_hours = st.slider(
            "Average Daily Electricity (hours/day)",
            min_value=1.0,
            max_value=18.0,
            value=6.0,
            step=0.5,
        )
        daily_data_mb = st.selectbox(
            "Typical Daily Data Budget (MB/day)",
            options=[150, 300, 500, 1000, 1500, 2500, 5000],
            index=2,
        )
        campus_lab_access = st.selectbox(
            "Campus Lab / Equipment Access",
            options=[0, 1, 2],
            format_func=lambda x: LAB_LABELS[x],
            index=0,
        )
        peer_group_score = st.slider(
            "Peer Study / Accountability Group (1 = Solo, 5 = Active Group)",
            min_value=1,
            max_value=5,
            value=2,
        )
        weekly_practice_hours = st.slider(
            "Current Weekly Practical / Study Hours",
            min_value=1.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
        )

        st.button("Diagnose Bottleneck & Generate Plan", type="primary", use_container_width=True)

    with col_output:
        st.subheader("2. Diagnostic & Self-Reliance Plan")
        input_df = pd.DataFrame([{
            "device_tier": device_tier,
            "daily_power_hours": daily_power_hours,
            "daily_data_mb": daily_data_mb,
            "campus_lab_access": campus_lab_access,
            "peer_group_score": peer_group_score,
            "weekly_practice_hours": weekly_practice_hours,
        }])

        pred_phase = int(model.predict(input_df)[0])
        probs = model.predict_proba(input_df)[0]
        phase_meta = PHASE_INFO[pred_phase]

        st.success(f"**Primary Recommended Phase:** {phase_meta['name']}")
        st.write(phase_meta["focus"])

        st.markdown("**Model Confidence Across Phases:**")
        prob_cols = st.columns(3)
        for idx, c in enumerate(prob_cols):
            short_title = PHASE_INFO[idx]["name"].split(":")[0]
            c.metric(short_title, f"{probs[idx]:.1%}")

        st.markdown("**Core Self-Reliance Strategies:**")
        for tip in phase_meta["actions"]:
            st.markdown(f"- {tip}")

        st.markdown(f"**7-Day Action Plan for *{goal}*:**")
        plan = generate_weekly_plan(
            goal=goal,
            device_tier=device_tier,
            power_hrs=daily_power_hours,
            data_mb=daily_data_mb,
            lab_access=campus_lab_access,
            peer_score=peer_group_score,
            practice_hrs=weekly_practice_hours,
            phase_id=pred_phase,
        )
        for day_title, day_desc in plan:
            st.markdown(f"**{day_title}:** {day_desc}")

        with st.expander("View Refined Prompt Engineering Template (Assignment 1 Link)"):
            st.code(
                build_refined_prompt(
                    goal=goal,
                    device_label=DEVICE_LABELS[device_tier],
                    power_hrs=daily_power_hours,
                    data_mb=daily_data_mb,
                    lab_label=LAB_LABELS[campus_lab_access],
                    peer_score=peer_group_score,
                    practice_hrs=weekly_practice_hours,
                    phase_name=phase_meta["name"],
                ),
                language="text",
            )

with tab2:
    st.subheader("Model Performance & Evaluation Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test Set Accuracy", f"{metrics['accuracy']:.2%}")
    m2.metric("Training Samples", metrics["train_samples"])
    m3.metric("Test Samples", metrics["test_samples"])

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Feature Importance (Random Forest)**")
        fi = pd.Series(metrics["feature_importances"]).sort_values(ascending=True)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        fi.plot(kind="barh", ax=ax1, color="#2b6cb0")
        ax1.set_xlabel("Importance Score")
        fig1.tight_layout()
        st.pyplot(fig1)

    with chart_col2:
        st.markdown("**Confusion Matrix (Actual vs. Predicted)**")
        cm = np.array(metrics["confusion_matrix"])
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        im = ax2.imshow(cm, cmap="Blues")
        ax2.set_xticks([0, 1, 2])
        ax2.set_yticks([0, 1, 2])
        ax2.set_xticklabels(["Phase 1", "Phase 2", "Phase 3"])
        ax2.set_yticklabels(["Phase 1", "Phase 2", "Phase 3"])
        ax2.set_xlabel("Predicted Phase")
        ax2.set_ylabel("Actual Phase")
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax2.text(c, r, str(cm[r, c]), ha="center", va="center", color="black")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("**Dataset Preview (`student_constraints.csv`)**")
    st.dataframe(df.head(15), use_container_width=True)
