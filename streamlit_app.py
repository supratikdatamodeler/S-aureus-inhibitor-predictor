from __future__ import annotations

# Streamlit entrypoint for Community Cloud deployment.
import html
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_ketcher import st_ketcher

from predictor import MAX_BATCH_BYTES, MODEL_METRICS, Predictor


st.set_page_config(
    page_title="SA-inhibitor-Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading the validated Bagging model…")
def load_predictor() -> Predictor:
    return Predictor()


def inject_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "style.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_validation_cards() -> None:
    st.markdown(
        f"""
        <div class="validation-stack">
          <div class="validation-card">
            <div class="validation-heading">
              <span>TRAINING PERFORMANCE</span><small>n = {MODEL_METRICS['training_samples']}</small>
            </div>
            <div class="validation-values">
              <div><strong>{MODEL_METRICS['train_accuracy']:.1%}</strong><span>Accuracy</span></div>
              <div><strong>{MODEL_METRICS['train_mcc']:.3f}</strong><span>MCC</span></div>
              <div><strong>{MODEL_METRICS['train_precision']:.1%}</strong><span>Precision</span></div>
              <div><strong>{MODEL_METRICS['train_recall']:.1%}</strong><span>Recall</span></div>
            </div>
          </div>
          <div class="validation-card test-card">
            <div class="validation-heading">
              <span>HELD-OUT TEST VALIDATION</span><small>n = {MODEL_METRICS['test_samples']}</small>
            </div>
            <div class="validation-values">
              <div><strong>{MODEL_METRICS['test_accuracy']:.1%}</strong><span>Accuracy</span></div>
              <div><strong>{MODEL_METRICS['test_mcc']:.3f}</strong><span>MCC</span></div>
              <div><strong>{MODEL_METRICS['test_precision']:.1%}</strong><span>Precision</span></div>
              <div><strong>{MODEL_METRICS['test_recall']:.1%}</strong><span>Recall</span></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_to_history(result: dict) -> None:
    st.session_state.history.insert(
        0,
        {
            "Canonical SMILES": result["canonical_smiles"],
            "Class": result["prediction_label"],
            "Higher-activity vote": f"{result['positive_vote']:.0%}",
            "Domain": result["applicability"]["label"],
        },
    )
    st.session_state.history = st.session_state.history[:10]


def render_single_result(result: dict) -> None:
    positive = result["predicted_class"] == 1
    badge_class = "positive" if positive else "negative"
    caption = (
        "Predicted above the dataset activity cutoff"
        if positive
        else "Predicted below the dataset activity cutoff"
    )
    domain = result["applicability"]
    st.markdown(
        f"""
        <div class="result-title">
          <div class="class-badge {badge_class}">{result['predicted_class']}</div>
          <div><small>PREDICTED CLASS</small><h3>{html.escape(result['prediction_label'])}</h3>
          <p>{caption}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_col, vote_col = st.columns([1, 1])
    with image_col:
        st.image(result["depiction_png"], use_container_width=True)
        st.code(result["canonical_smiles"], language=None)
    with vote_col:
        st.metric("Higher-activity vote", f"{result['positive_vote']:.0%}")
        st.progress(result["positive_vote"])
        st.caption("Ensemble support from 20 trees; not a calibrated probability.")
        st.markdown(
            f'<span class="domain-badge {domain["level"]}">{domain["label"]}</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"{domain['range_fraction']:.0%} of descriptors within training ranges · "
            f"nearest distance {domain['nearest_distance']} "
            f"(95% threshold {domain['distance_threshold']})"
        )

    property_items = list(result["properties"].items())
    for row_start in (0, 3):
        property_columns = st.columns(3)
        for column, (name, value) in zip(
            property_columns, property_items[row_start : row_start + 3]
        ):
            column.metric(name, value)

    st.markdown("#### Largest descriptor deviations")
    st.dataframe(
        pd.DataFrame(domain["top_deviations"]),
        hide_index=True,
        use_container_width=True,
    )


def render_batch_result(batch: dict) -> None:
    summary = batch["summary"]
    st.markdown("### Batch prediction complete")
    st.caption(
        f"{batch['filename']} · {summary['successful']} successful of {summary['total']}"
    )
    columns = st.columns(4)
    for column, label, value in zip(
        columns,
        ["Total", "Higher activity", "Inside domain", "Failed"],
        [
            summary["total"],
            summary["higher_activity"],
            summary["inside_domain"],
            summary["failed"],
        ],
    ):
        column.metric(label, value)

    frame = pd.DataFrame(batch["rows"])
    display_columns = [
        column
        for column in [
            "index",
            "name",
            "canonical_smiles",
            "predicted_class",
            "higher_activity_vote",
            "domain",
            "status",
            "error",
        ]
        if column in frame.columns
    ]
    st.dataframe(frame[display_columns], hide_index=True, use_container_width=True)
    st.download_button(
        "Download prediction CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="sa_inhibitor_batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


inject_css()
predictor = load_predictor()

for key, default in {
    "single_result": None,
    "batch_result": None,
    "history": [],
    "single_error": "",
    "batch_error": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(
    """
    <div class="top-brand">
      <span>SA</span>
      <div><strong>SA-inhibitor-Predictor</strong><small>Structure-based ML screening</small></div>
      <b>MODEL READY</b>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.5, 1], gap="large")
with hero_left:
    st.markdown('<p class="eyebrow">STRUCTURE-BASED ML SCREENING</p>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="hero-title">Prioritize candidate<br><em>S. aureus</em> inhibitors.</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-copy">Draw a molecule, paste SMILES, or upload a structure batch. '
        "The supplied 20-tree Bagging classifier evaluates 24 validated Mordred "
        "descriptors and reports activity class with applicability-domain context.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pipeline"><span>Structure</span><b>→</b><span>24 descriptors</span>'
        '<b>→</b><span>Bagging vote</span></div>',
        unsafe_allow_html=True,
    )
with hero_right:
    render_validation_cards()

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
input_col, result_col = st.columns(2, gap="large")

with input_col:
    with st.container(border=True):
        st.markdown("### 01 · Define structure")
        mode = st.segmented_control(
            "Input method",
            ["Draw structure", "Paste SMILES", "Batch upload"],
            default="Draw structure",
            label_visibility="collapsed",
        )

        if mode == "Draw structure":
            drawn_smiles = st_ketcher("")
            if st.button("Run prediction", type="primary", use_container_width=True):
                try:
                    with st.spinner("Calculating 24 molecular descriptors…"):
                        st.session_state.single_result = predictor.predict(
                            drawn_smiles or ""
                        )
                    st.session_state.batch_result = None
                    st.session_state.single_error = ""
                    add_to_history(st.session_state.single_result)
                except Exception as exc:
                    st.session_state.single_error = str(exc)

        elif mode == "Paste SMILES":
            smiles = st.text_area(
                "SMILES",
                placeholder="Example: CC(=O)Oc1ccccc1C(=O)O",
                height=150,
            )
            example_col1, example_col2 = st.columns(2)
            if example_col1.button("Use nalidixic acid", use_container_width=True):
                smiles = "CCn1cc(C(=O)O)c(=O)c2ccc(C)nc21"
            if example_col2.button("Use aspirin", use_container_width=True):
                smiles = "CC(=O)Oc1ccccc1C(=O)O"
            if st.button("Run prediction", type="primary", use_container_width=True):
                try:
                    with st.spinner("Calculating 24 molecular descriptors…"):
                        st.session_state.single_result = predictor.predict(smiles)
                    st.session_state.batch_result = None
                    st.session_state.single_error = ""
                    add_to_history(st.session_state.single_result)
                except Exception as exc:
                    st.session_state.single_error = str(exc)

        else:
            uploaded = st.file_uploader(
                "Compound structure file",
                type=["smi", "smiles", "txt", "csv", "sdf"],
                help="Maximum 100 structures and 5 MB per file.",
            )
            st.caption(
                "SMI/TXT: one SMILES per line with optional name · "
                "CSV: a column containing 'SMILES' · SDF: one or more records"
            )
            if st.button(
                "Run batch prediction",
                type="primary",
                use_container_width=True,
                disabled=uploaded is None,
            ):
                try:
                    if uploaded.size > MAX_BATCH_BYTES:
                        raise ValueError("The uploaded file exceeds the 5 MB limit.")
                    text = uploaded.getvalue().decode("utf-8", errors="replace")
                    with st.spinner("Processing the structure batch…"):
                        st.session_state.batch_result = predictor.predict_batch(
                            uploaded.name, text
                        )
                    st.session_state.single_result = None
                    st.session_state.batch_error = ""
                except Exception as exc:
                    st.session_state.batch_error = str(exc)

        if st.session_state.single_error:
            st.error(st.session_state.single_error)
        if st.session_state.batch_error:
            st.error(st.session_state.batch_error)

with result_col:
    with st.container(border=True):
        st.markdown("### 02 · Prediction")
        st.caption("RESEARCH USE ONLY")
        if st.session_state.batch_result:
            render_batch_result(st.session_state.batch_result)
        elif st.session_state.single_result:
            render_single_result(st.session_state.single_result)
        else:
            st.markdown(
                """
                <div class="empty-result">
                  <div>◌</div><h3>Awaiting a structure</h3>
                  <p>Classification, ensemble support, molecular properties and
                  domain assessment will appear here.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
st.markdown('<p class="eyebrow">MODEL CARD</p>', unsafe_allow_html=True)
st.markdown("## What this result means")
st.write(
    "The model separates the supplied balanced dataset at approximately pIC50 4.97. "
    "Higher activity means class 1 in that dataset; it does not establish MIC, "
    "bactericidal activity, safety, or efficacy."
)
model_columns = st.columns(4)
for column, title, value, description in [
    (
        model_columns[0],
        "DATA",
        "474 / 158",
        "Training / held-out test compounds",
    ),
    (
        model_columns[1],
        "CLASSIFICATION",
        "20 trees",
        "Bagging ensemble with balanced labels",
    ),
    (
        model_columns[2],
        "FEATURES",
        "24",
        "Selected Mordred 2D descriptors",
    ),
    (
        model_columns[3],
        "DOMAIN",
        "k-NN + range",
        "Standardized descriptor-space screening",
    ),
]:
    with column:
        with st.container(border=True):
            st.caption(title)
            st.markdown(f"### {value}")
            st.caption(description)

st.warning(
    "Use predictions to prioritize compounds for experimental testing. "
    "Do not use this classifier for clinical decisions or claims of antimicrobial efficacy."
)

if st.session_state.history:
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown('<p class="eyebrow">SESSION LOG</p>', unsafe_allow_html=True)
    st.markdown("## Recent predictions")
    st.dataframe(
        pd.DataFrame(st.session_state.history),
        hide_index=True,
        use_container_width=True,
    )

st.markdown(
    '<div class="footer">SA-inhibitor-Predictor · BaggingClassifier · '
    "scikit-learn 1.6.1 · RDKit + Mordred</div>",
    unsafe_allow_html=True,
)
