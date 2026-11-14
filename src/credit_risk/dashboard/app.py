"""Streamlit dashboard for loan officers. Talks to the FastAPI service only (API_URL)."""

from __future__ import annotations

import math
import os
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", API_URL).rstrip("/")

st.set_page_config(page_title="Credit risk: loan officer view", page_icon=":bank:", layout="wide")


# ---------------------------------------------------------------- API helpers
@st.cache_resource
def client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=30)


@st.cache_data(ttl=300)
def get(path: str) -> Any:
    r = client().get(path)
    r.raise_for_status()
    return r.json()


def predict(applicant: dict[str, Any]) -> dict[str, Any]:
    clean = {k: v for k, v in applicant.items() if not (isinstance(v, float) and math.isnan(v))}
    r = client().post("/predict", params={"explain": "true"}, json=clean)
    if r.status_code == 422:
        detail = r.json().get("detail", [])
        msgs = "; ".join(
            f"{'.'.join(str(x) for x in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail
        )
        raise ValueError(msgs or r.text)
    r.raise_for_status()
    return r.json()


try:
    MODEL = get("/model")
    SCHEMA = {f["name"]: f for f in get("/schema")}
    SAMPLES = get("/samples")
except httpx.HTTPError as exc:
    st.error(f"Can't reach the scoring API at {API_URL} ({exc}). Is it running?")
    st.stop()

THRESHOLD = float(MODEL["threshold"])
SEGMENT_LABEL = {
    "thin_file": "thin-file (no bureau, no previous applications)",
    "no_bureau_has_prev": "no bureau record, has Home Credit history",
    "has_bureau": "has a credit-bureau record",
}


def stat(name: str, key: str, default: float) -> float:
    v = SCHEMA[name]["stats"].get(key)
    return default if v is None else float(v)


def fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    st.markdown(
        f"Version `{MODEL['version']}`  \n"
        f"Decline when P(default) ≥ **{THRESHOLD:.3f}**  \n"
        f"Cost assumptions: loss on default {MODEL['cost_assumptions']['loss_given_default']:.0%} of the loan, "
        f"lost margin on a declined good customer {MODEL['cost_assumptions']['margin_good_loan']:.0%}."
    )
    m = MODEL["holdout_metrics"]
    st.caption(
        f"Hold-out ROC-AUC: all {m['overall']['roc_auc']:.3f} · thin-file {m['thin_file']['roc_auc']:.3f} "
        f"(n={m['thin_file']['n']})"
    )
    st.divider()
    mode = st.radio("Applicant", ["Sample applicant", "New applicant (form)"], index=0)
    st.caption(
        "Sample applicants are synthetic, generated from the training data's distributions; they are not real people."
    )
    st.caption(f"API: [{PUBLIC_API_URL}]({PUBLIC_API_URL}/docs) (interactive docs)")


# ---------------------------------------------------------------- applicant input
def form_applicant() -> dict[str, Any]:
    st.subheader("New applicant")
    st.caption(
        "A new customer with no credit-bureau record and no earlier Home Credit loans, i.e. a first-time borrower. History fields are left empty."
    )

    def cat(name: str, label: str, col) -> str | None:
        options = ["(not given)", *SCHEMA[name]["categories"]]
        top = SCHEMA[name]["stats"].get("top", [None])[0]
        idx = options.index(top) if top in options else 0
        choice = col.selectbox(label, options, index=idx, key=f"form_{name}")
        return None if choice == "(not given)" else choice

    c1, c2, c3 = st.columns(3)
    a: dict[str, Any] = {}
    a["NAME_CONTRACT_TYPE"] = cat("NAME_CONTRACT_TYPE", "Contract type", c1)
    a["AMT_CREDIT"] = c1.number_input(
        "Loan amount", min_value=1000.0, value=stat("AMT_CREDIT", "p50", 500000), step=10000.0
    )
    a["AMT_GOODS_PRICE"] = c1.number_input(
        "Price of goods financed",
        min_value=0.0,
        value=stat("AMT_GOODS_PRICE", "p50", 450000),
        step=10000.0,
    )
    a["AMT_ANNUITY"] = c1.number_input(
        "Annuity (regular repayment)",
        min_value=0.0,
        value=stat("AMT_ANNUITY", "p50", 25000),
        step=1000.0,
    )
    a["AMT_INCOME_TOTAL"] = c2.number_input(
        "Annual income",
        min_value=1000.0,
        value=stat("AMT_INCOME_TOTAL", "p50", 150000),
        step=5000.0,
    )
    a["NAME_INCOME_TYPE"] = cat("NAME_INCOME_TYPE", "Income type", c2)
    a["OCCUPATION_TYPE"] = cat("OCCUPATION_TYPE", "Occupation", c2)
    a["ORGANIZATION_TYPE"] = cat("ORGANIZATION_TYPE", "Employer type", c2)
    a["YEARS_EMPLOYED"] = c2.number_input(
        "Years in current job",
        min_value=0.0,
        max_value=50.0,
        value=stat("YEARS_EMPLOYED", "p50", 3.0),
        step=0.5,
    )
    a["NAME_EDUCATION_TYPE"] = cat("NAME_EDUCATION_TYPE", "Education", c3)
    a["NAME_HOUSING_TYPE"] = cat("NAME_HOUSING_TYPE", "Housing", c3)
    a["FLAG_OWN_REALTY"] = float(c3.checkbox("Owns property", value=True))
    a["FLAG_OWN_CAR"] = float(c3.checkbox("Owns a car", value=False))
    a["REGION_RATING_CLIENT"] = float(
        c3.selectbox("Region rating (1 best - 3 worst)", [1, 2, 3], index=1)
    )
    a["REGION_RATING_CLIENT_W_CITY"] = a["REGION_RATING_CLIENT"]

    with st.expander("Identity, contact and external scores"):
        d1, d2, d3 = st.columns(3)
        a["YEARS_ID_PUBLISH"] = d1.number_input(
            "Years since ID document changed", 0.0, 60.0, stat("YEARS_ID_PUBLISH", "p50", 8.0), 0.5
        )
        a["YEARS_REGISTRATION"] = d1.number_input(
            "Years since registration changed",
            0.0,
            70.0,
            stat("YEARS_REGISTRATION", "p50", 12.0),
            0.5,
        )
        a["YEARS_SINCE_PHONE_CHANGE"] = d1.number_input(
            "Years since phone changed",
            0.0,
            12.0,
            stat("YEARS_SINCE_PHONE_CHANGE", "p50", 2.0),
            0.25,
        )
        a["FLAG_EMP_PHONE"] = float(d2.checkbox("Gave an employer phone", value=True))
        a["FLAG_WORK_PHONE"] = float(d2.checkbox("Gave a work phone", value=False))
        a["FLAG_PHONE"] = float(d2.checkbox("Gave a home phone", value=False))
        a["FLAG_EMAIL"] = float(d2.checkbox("Gave an email", value=False))
        a["FLAG_DOCUMENT_3"] = float(d2.checkbox("Provided document 3", value=True))
        a["DOCUMENT_COUNT"] = float(d2.number_input("Documents provided", 0, 10, 1))
        for i in (1, 2, 3):
            name = f"EXT_SOURCE_{i}"
            known = d3.checkbox(f"External score {i} available", value=i != 1, key=f"known_{name}")
            a[name] = (
                d3.slider(
                    f"External score {i}",
                    0.0,
                    1.0,
                    stat(name, "p50", 0.5),
                    0.01,
                    disabled=not known,
                )
                if known
                else None
            )
        for flag in (
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY",
        ):
            a[flag] = 0.0
        a["REGION_POPULATION_RELATIVE"] = stat("REGION_POPULATION_RELATIVE", "p50", 0.02)
        a["NAME_TYPE_SUITE"] = "Unaccompanied"
        for name in (
            "OBS_30_CNT_SOCIAL_CIRCLE",
            "DEF_30_CNT_SOCIAL_CIRCLE",
            "OBS_60_CNT_SOCIAL_CIRCLE",
            "DEF_60_CNT_SOCIAL_CIRCLE",
        ):
            a[name] = 0.0
    # No bureau record and no previous applications -> thin-file by definition.
    a["BUR_COUNT"] = 0.0
    a["PREV_COUNT"] = 0.0
    return {k: v for k, v in a.items() if v is not None}


def sample_applicant() -> tuple[dict[str, Any], str]:
    st.subheader("Sample applicant")
    seg = st.selectbox("Segment", list(SEGMENT_LABEL), format_func=lambda s: SEGMENT_LABEL[s])
    rows = [s for s in SAMPLES if s["segment"] == seg]
    ids = [r["id"] for r in rows]
    chosen = st.selectbox("Applicant (synthetic)", ids)
    row = next(r for r in rows if r["id"] == chosen)
    return dict(row["applicant"]), chosen


if mode == "Sample applicant":
    applicant, applicant_id = sample_applicant()
else:
    applicant, applicant_id = form_applicant(), "form"

# ---------------------------------------------------------------- what-if sliders
WHAT_IF = [
    ("AMT_CREDIT", "Loan amount", 1000.0),
    ("AMT_ANNUITY", "Annuity", 100.0),
    ("AMT_INCOME_TOTAL", "Annual income", 1000.0),
    ("YEARS_EMPLOYED", "Years in current job", 0.1),
    ("EXT_SOURCE_2", "External score 2", 0.01),
    ("EXT_SOURCE_3", "External score 3", 0.01),
]

with st.expander("What-if: change key inputs and re-score", expanded=True):
    st.caption(
        "Sliders start at this applicant's values. Ratio features (loan/income, annuity/income, ...) are recomputed by the API."
    )
    adjusted = dict(applicant)
    cols = st.columns(3)
    for i, (name, label, step) in enumerate(WHAT_IF):
        current = applicant.get(name)
        lo, hi = stat(name, "p01", 0.0), stat(name, "p99", 1.0)
        if current is None:
            cols[i % 3].caption(f"{label}: not provided for this applicant")
            continue
        lo, hi = min(lo, float(current)), max(hi, float(current))
        adjusted[name] = cols[i % 3].slider(
            label,
            float(lo),
            float(hi),
            float(current),
            float(step),
            key=f"wi_{applicant_id}_{name}",
        )

# ---------------------------------------------------------------- score
try:
    base = predict(applicant)
    result = predict(adjusted) if adjusted != applicant else base
except ValueError as exc:
    st.error(f"The API rejected the input: {exc}")
    st.stop()

prob = result["probability"]
left, right = st.columns([1, 2])
with left:
    decision = result["decision"]
    colour = "#1a7f37" if decision == "approve" else "#c0392b"
    st.markdown(
        f"<div style='padding:14px;border-radius:8px;background:{colour};color:white;font-size:1.6em;"
        f"text-align:center'>{decision.upper()}</div>",
        unsafe_allow_html=True,
    )
    delta = None if result is base else f"{(prob - base['probability']) * 100:+.1f} pts vs original"
    st.metric("Probability of default", fmt_pct(prob), delta=delta, delta_color="inverse")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": colour},
                "threshold": {"line": {"color": "black", "width": 3}, "value": THRESHOLD * 100},
            },
        )
    )
    gauge.update_layout(height=220, margin={"l": 20, "r": 20, "t": 10, "b": 10})
    st.plotly_chart(gauge, width="stretch")
    st.caption(f"Black line = decline threshold ({fmt_pct(THRESHOLD)}).")
    if result["thin_file"]:
        st.info(
            "Thin-file applicant: no credit-bureau record and no earlier Home Credit applications."
        )
    for w in result.get("warnings", []):
        st.warning(w)

with right:
    st.markdown("**Top reasons raising the risk**")
    for r in result["reasons"] or [{"text": "None: every factor group lowered the risk."}]:
        st.markdown(f"- {r['text']}")
    st.markdown("**Factors lowering the risk**")
    for r in result["strengths"]:
        st.markdown(f"- {r['text']}")

    expl = result["explanation"]
    contrib = pd.Series(expl["contributions"])
    labels = {n: SCHEMA[n]["label"] for n in contrib.index}
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:12]
    rest = contrib.drop(top.index).sum()
    names = [labels[n] for n in top.index][::-1]
    values = top.to_numpy()[::-1].tolist()
    names = ["Other features", *names]
    values = [rest, *values]
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=["Average applicant", *names, "This applicant"],
            x=[expl["base_value"], *values, 0],
            measure=["absolute", *["relative"] * len(values), "total"],
            increasing={"marker": {"color": "#c0392b"}},
            decreasing={"marker": {"color": "#1a7f37"}},
            totals={"marker": {"color": "#1f4e79"}},
        )
    )
    fig.update_layout(
        title="SHAP waterfall: red raises risk, green lowers it",
        height=520,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis_title="log-odds of default",
    )
    st.plotly_chart(fig, width="stretch")

with st.expander("All inputs sent to the model"):
    vals = expl["values"]
    table = pd.DataFrame(
        [
            {
                "feature": n,
                "meaning": SCHEMA[n]["label"],
                "value": vals.get(n),
                "SHAP": expl["contributions"][n],
                "derived": SCHEMA[n]["derived"],
            }
            for n in SCHEMA
        ]
    )
    table["value"] = table["value"].astype(str)
    st.dataframe(table, hide_index=True, width="stretch")
