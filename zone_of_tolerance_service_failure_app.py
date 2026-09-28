"""Streamlit app comparing baseline CLV with an optional service-failure scenario."""
from __future__ import annotations

import asyncio
import hashlib
import random
import re
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
DEFAULT_USERS_FILE = APP_DIR / "users.xlsx"
MODEL_NAME = "gemini-3.1-flash-lite"
CAC_PER_CUSTOMER = 100.0
SIMULATION_VERSION = 4
YEARS = range(1, 6)


class RetentionResponse(BaseModel):
    year_1_probability: int = Field(ge=0, le=100)
    year_2_probability: int = Field(ge=0, le=100)
    year_3_probability: int = Field(ge=0, le=100)
    year_4_probability: int = Field(ge=0, le=100)
    year_5_probability: int = Field(ge=0, le=100)
    reason: str = Field(description="One concise sentence explaining the decisions")


class DissatisfactionResponse(BaseModel):
    reason: str = Field(
        description="One concise, realistic first-person reason why this customer became dissatisfied"
    )


def clean_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


@st.cache_data(show_spinner=False)
def load_default_profiles() -> pd.DataFrame:
    return pd.read_excel(DEFAULT_USERS_FILE)


def load_profiles(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return load_default_profiles()
    if Path(uploaded_file.name).suffix.lower() == ".csv":
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def matching_column(dataframe: pd.DataFrame, wanted: str) -> str | None:
    names = {str(column).strip().casefold(): str(column) for column in dataframe.columns}
    return names.get(wanted.casefold())


def age_group_for(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    age = pd.to_numeric(value, errors="coerce")
    if pd.isna(age):
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if not numbers:
            return text or None
        age = float(numbers[0])
    if age < 18:
        return "Under 18"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    return "65+"


def filter_profiles(dataframe: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Choose customer profiles")
    fields = (
        ("Age group", matching_column(dataframe, "Age"), age_group_for),
        ("Location", matching_column(dataframe, "Location"), clean_value),
        ("Gender", matching_column(dataframe, "Gender"), clean_value),
    )
    mask = pd.Series(True, index=dataframe.index)
    for container, (label, source, transform) in zip(st.columns(3), fields):
        if not source:
            container.caption(f"{label} filter unavailable.")
            continue
        values = dataframe[source].map(transform)
        options = sorted(set(values.dropna()), key=lambda item: str(item).casefold())
        selected = container.multiselect(label, options, placeholder=f"All {label.lower()}s")
        if selected:
            mask &= values.isin(selected)

    matches = dataframe.loc[mask].copy()
    if matches.empty:
        st.warning("No profiles match the selected filters.")
        return matches
    count = int(st.number_input(
        "Number of profiles to simulate", 1, len(matches), len(matches), 1,
        help="A reproducible random sample is used when fewer profiles are selected.",
    ))
    if count < len(matches):
        matches = matches.sample(count, random_state=42).sort_index()
    return matches.reset_index(drop=True)


def configured_api_key() -> str:
    try:
        return str(st.secrets["GOOGLE_API_KEY"])
    except (KeyError, FileNotFoundError):
        return ""


def make_prompt(firm: str, description: str, persona: str, profile: dict[str, str]) -> str:
    profile_text = "\n".join(f"- {key}: {value}" for key, value in profile.items())
    return f"""Act as a digital twin of a potential customer.

Firm: {firm}
Firm/service description: {description}
Target customer persona supplied by the researcher:
{persona}

Individual customer profile:
{profile_text}

Estimate this customer's probability (an integer from 0 to 100) of using the service again in
each of the next five years. Years 2-5 are conditional probabilities: estimate the chance of
returning that year if the customer was still active in the prior year.

Be realistically calibrated, not promotional. Account for ordinary churn, changing needs,
price sensitivity, competing services, relocation, and declining relevance when consistent
with the supplied profile. Do not invent discounts, product changes, or personal facts.
Return five probabilities and one concise reason."""


def stable_number(*parts: object) -> float:
    key = "|".join(str(part).casefold().strip() for part in parts).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return integer / (2**64 - 1)


async def query_profile(profile_id: int, firm: str, prompt: str, llm, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        for attempt in range(5):
            try:
                response = await llm.ainvoke(prompt)
                retained = True
                decisions = {}
                for year in YEARS:
                    probability = int(getattr(response, f"year_{year}_probability"))
                    decisions[f"Year {year} probability"] = probability / 100
                    retained = retained and stable_number(firm, profile_id, year) * 100 < probability
                    decisions[f"Year {year} return"] = retained
                return {"Profile ID": profile_id, **decisions, "Reason": response.reason, "Status": "Success"}
            except Exception as exc:
                if attempt < 4:
                    await asyncio.sleep(2**attempt)
                else:
                    return {"Profile ID": profile_id, "Status": f"Error: {exc}"}
    return {"Profile ID": profile_id, "Status": "Unknown error"}


async def run_all(tasks: list, progress, item_label: str = "profiles simulated") -> list[dict]:
    results = []
    for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
        results.append(await task)
        progress.progress(completed / len(tasks), text=f"{completed} of {len(tasks)} {item_label}")
    return results


def simulate_retention(profiles_df: pd.DataFrame, firm: str, description: str, persona: str, api_key: str) -> pd.DataFrame:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, temperature=0.5, google_api_key=api_key
    ).with_structured_output(RetentionResponse)
    semaphore = asyncio.Semaphore(40)
    profiles = {}
    tasks = []
    for profile_id, (_, row) in enumerate(profiles_df.iterrows(), start=1):
        profile = {str(column): value for column, raw in row.items() if (value := clean_value(raw)) is not None}
        profiles[profile_id] = profile
        tasks.append(query_profile(profile_id, firm, make_prompt(firm, description, persona, profile), llm, semaphore))
    progress = st.progress(0, text="Preparing profile simulations...")
    responses = asyncio.run(run_all(tasks, progress))
    progress.empty()
    records = [{**profiles[item["Profile ID"]], **item} for item in responses]
    return pd.DataFrame(records).sort_values("Profile ID")


def calculate_clv(results: pd.DataFrame, monthly_sales: float, margin_percent: float) -> pd.DataFrame:
    successful = results.loc[results["Status"] == "Success"]
    annual_sales = monthly_sales * 12
    margin = margin_percent / 100
    return pd.DataFrame([
        {
            "Year": f"Year {year}",
            "Returning profiles": (retained := int(successful[f"Year {year} return"].sum())),
            "Retention rate": retained / len(successful),
            "Annual sales per profile": annual_sales,
            "Margin": margin,
            "Annual CLV contribution": annual_sales * retained * margin,
        }
        for year in YEARS
    ])


def failure_prompt(firm: str, description: str, year: int, row: pd.Series) -> str:
    excluded = {"Profile ID", "Status", "Reason"}
    profile = {
        str(column): clean_value(value)
        for column, value in row.items()
        if column not in excluded and not str(column).startswith("Year ") and clean_value(value) is not None
    }
    profile_text = "\n".join(f"- {key}: {value}" for key, value in profile.items()) or "- No additional profile fields"
    return f"""Act as this customer's digital twin in a fictional service-failure exercise.

Firm: {firm}
Firm/service description: {description}
The service failure occurs in Year {year}.
Customer profile:
{profile_text}
Baseline retention rationale: {row.get('Reason', '')}

Generate one plausible and specific reason this customer became angry or dissatisfied with the
service in that year. Make it relevant to the offering and, where useful, the profile. Use a
short first-person sentence. Do not mention that this is a simulation, do not blame the customer,
and do not invent severe injury, crime, discrimination, or other extreme events."""


async def query_failure_reason(profile_id: int, year: int, prompt: str, llm, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        for attempt in range(5):
            try:
                response = await llm.ainvoke(prompt)
                return {"Profile ID": profile_id, "Failure year": f"Year {year}", "Dissatisfaction reason": response.reason}
            except Exception as exc:
                if attempt < 4:
                    await asyncio.sleep(2**attempt)
                else:
                    return {
                        "Profile ID": profile_id,
                        "Failure year": f"Year {year}",
                        "Dissatisfaction reason": f"AI reason unavailable: {exc}",
                    }
    return {}


def simulate_service_failures(
    baseline: pd.DataFrame, angry_percent: float, firm: str, description: str, api_key: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly select angry customers and failure years, then generate their reasons with AI."""
    scenario = baseline.copy()
    successful = scenario.loc[scenario["Status"] == "Success"].copy()
    requested = min(len(successful), max(1, round(len(successful) * angry_percent / 100)))
    chosen_ids = random.sample(successful["Profile ID"].tolist(), requested)

    assignments = []
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, temperature=0.8, google_api_key=api_key
    ).with_structured_output(DissatisfactionResponse)
    semaphore = asyncio.Semaphore(40)
    tasks = []
    for profile_id in chosen_ids:
        row = successful.loc[successful["Profile ID"] == profile_id].iloc[0]
        # Assign every selected angry customer directly across all five years.
        year = random.randint(1, 5)
        assignments.append((profile_id, year))
        tasks.append(query_failure_reason(
            profile_id, year, failure_prompt(firm, description, year, row), llm, semaphore
        ))

    progress = st.progress(0, text="Generating angry-customer reasons...")
    generated = asyncio.run(run_all(tasks, progress, "AI dissatisfaction reasons generated"))
    progress.empty()
    assignment_frame = pd.DataFrame(generated)

    for profile_id, year in assignments:
        row_mask = scenario["Profile ID"] == profile_id
        reason_match = assignment_frame.loc[assignment_frame["Profile ID"] == profile_id, "Dissatisfaction reason"]
        reason = reason_match.iloc[0] if not reason_match.empty else "AI reason unavailable"
        for later_year in range(year, 6):
            scenario.loc[row_mask, f"Year {later_year} return"] = False
        scenario.loc[row_mask, "Service failure year"] = f"Year {year}"
        scenario.loc[row_mask, "Dissatisfaction reason"] = reason

    return scenario, assignment_frame


def recovery_suggestion(reason: str) -> str:
    text = reason.casefold()
    if any(word in text for word in ("wait", "slow", "delay", "late")):
        return "Set a response-time standard, warn customers early about delays, and offer a fast recovery option."
    if any(word in text for word in ("price", "cost", "fee", "expensive", "billing")):
        return "Make pricing and fees transparent, audit billing errors, and offer a fair-value recovery when the firm is at fault."
    if any(word in text for word in ("rude", "staff", "employee", "support", "service")):
        return "Coach frontline staff on empathy and ownership, then empower them to resolve the issue in one interaction."
    if any(word in text for word in ("quality", "broken", "defect", "reliable")):
        return "Identify the quality failure, correct the process causing it, and proactively replace or redo the service."
    if any(word in text for word in ("app", "website", "technology", "login", "technical")):
        return "Remove the digital friction, provide an immediate assisted channel, and notify affected customers when fixed."
    return "Acknowledge the specific failure, explain the remedy, provide proportionate recovery, and confirm that the fix worked."


def comparison_figure(baseline: pd.DataFrame, failure: pd.DataFrame, firm: str):
    figure, axis = plt.subplots(figsize=(9, 5.2))
    x = list(range(5))
    width = 0.36
    axis.bar([value - width / 2 for value in x], baseline["Annual CLV contribution"], width, label="No service failure", color="#2563eb")
    axis.bar([value + width / 2 for value in x], failure["Annual CLV contribution"], width, label="With service failure", color="#dc2626")
    axis.set_xticks(x, baseline["Year"])
    axis.set_title(f"CLV impact of service failure — {firm}", pad=14)
    axis.set_ylabel("CLV contribution ($)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_") or "firm"


st.set_page_config(page_title="CLV Service-Failure Simulator", page_icon="📉", layout="wide")
st.title("Customer Lifetime Value and Service Failure Simulator")
st.caption("First establish baseline CLV:CAC. Then add an optional service-failure scenario and compare the impact.")

api_key = configured_api_key()
if not api_key:
    st.error("Gemini API key is not configured. Add GOOGLE_API_KEY to the app's Streamlit secrets.")
    st.stop()

st.subheader("1. Firm and customer assumptions")
firm_name = st.text_input("Firm name", placeholder="e.g., Acme Fitness")
firm_description = st.text_area("Firm and service description", height=150)
persona = st.text_area("Target customer persona", height=220)
sales_col, margin_col = st.columns(2)
monthly_sales = sales_col.number_input("Expected monthly sales per returning customer ($)", min_value=0.0, value=100.0, step=10.0)
margin_percent = margin_col.number_input("Profit margin (%)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

fixed1, fixed2, fixed3 = st.columns(3)
fixed1.metric("Customer acquisition cost", "$100 per customer")
fixed2.metric("Simulation horizon", "5 years")
fixed3.metric("Annual sales basis", f"${monthly_sales * 12:,.2f} per customer")
st.info("CAC is fixed at $100 per acquired customer. Monthly sales are annualized (× 12); no discount rate is applied.")

with st.expander("Customer profile source"):
    uploaded_file = st.file_uploader("Optional profile file", type=["xlsx", "xls", "csv"])

try:
    all_profiles = load_profiles(uploaded_file).dropna(how="all").reset_index(drop=True)
    selected_profiles = filter_profiles(all_profiles)
    st.caption(f"{len(selected_profiles):,} of {len(all_profiles):,} profiles selected")
    st.dataframe(selected_profiles, use_container_width=True, height=260)
except Exception as exc:
    st.error(f"Could not load customer profiles: {exc}")
    st.stop()

if st.button("Run baseline: no service failure", type="primary", use_container_width=True):
    missing = [label for label, value in (
        ("firm name", firm_name), ("firm and service description", firm_description),
        ("target customer persona", persona),
    ) if not value.strip()]
    if missing:
        st.error("Please provide: " + ", ".join(missing) + ".")
    elif selected_profiles.empty:
        st.error("Select at least one customer profile.")
    else:
        with st.spinner("Simulating baseline repeat use across customer profiles..."):
            responses = simulate_retention(selected_profiles, firm_name, firm_description, persona, api_key)
        st.session_state["clv_run_service_failure"] = {
            "responses": responses, "firm": firm_name, "monthly_sales": monthly_sales,
            "margin": margin_percent, "description": firm_description, "version": SIMULATION_VERSION,
        }
        st.session_state.pop("service_failure_result", None)

state_key = "clv_run_service_failure"
if state_key in st.session_state and st.session_state[state_key].get("version") != SIMULATION_VERSION:
    del st.session_state[state_key]
    st.info("The simulation method was updated. Run the baseline again.")

if state_key in st.session_state:
    run = st.session_state[state_key]
    results = run["responses"]
    successful = results.loc[results["Status"] == "Success"]
    st.divider()
    st.subheader("Baseline results: no service failure")
    if successful.empty:
        st.error("No simulations succeeded. Review the Status column.")
        st.dataframe(results, use_container_width=True)
        st.stop()

    baseline_summary = calculate_clv(results, run["monthly_sales"], run["margin"])
    baseline_total = float(baseline_summary["Annual CLV contribution"].sum())
    acquired = len(successful)
    baseline_average = baseline_total / acquired
    baseline_ratio = baseline_average / CAC_PER_CUSTOMER
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("5-year cohort CLV", f"${baseline_total:,.2f}")
    b2.metric("Average CLV per profile", f"${baseline_average:,.2f}")
    b3.metric("Total acquisition cost", f"${acquired * CAC_PER_CUSTOMER:,.2f}")
    b4.metric("Baseline CLV : CAC", f"{baseline_ratio:.2f} : 1")

    st.subheader("2. Optional service-failure scenario")
    st.caption(
        "Enter only the angry-customer proportion. The app randomly selects that proportion of customers, "
        "randomly assigns them across Years 1–5, "
        "then each AI customer explains why they became dissatisfied."
    )
    angry_percent = st.slider(
        "What proportion of customers should become angry?", 0, 100, 0, 1,
        help="Selected customers stop returning in their assigned failure year and all later years.",
    )

    if angry_percent > 0:
        expected_angry = max(1, round(acquired * angry_percent / 100))
        st.info(f"The scenario will create {expected_angry:,} angry AI customer(s) from {acquired:,} successful profiles.")
        if st.button("Run service-failure simulation", type="primary", use_container_width=True):
            with st.spinner("Randomly assigning angry customers across five years and generating their reasons..."):
                scenario, assignments = simulate_service_failures(
                    results, angry_percent, run["firm"], run["description"], api_key
                )
            st.session_state["service_failure_result"] = {
                "scenario": scenario, "assignments": assignments, "angry_percent": angry_percent,
                "baseline_fingerprint": id(results),
            }

    scenario_run = st.session_state.get("service_failure_result")
    if scenario_run and scenario_run.get("baseline_fingerprint") == id(results):
        scenario = scenario_run["scenario"]
        assignments = scenario_run["assignments"]
        failure_summary = calculate_clv(scenario, run["monthly_sales"], run["margin"])
        failure_total = float(failure_summary["Annual CLV contribution"].sum())
        failure_average = failure_total / acquired
        failure_ratio = failure_average / CAC_PER_CUSTOMER
        ratio_drop = baseline_ratio - failure_ratio

        st.divider()
        st.subheader("Service-failure impact")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CLV:CAC with failure", f"{failure_ratio:.2f} : 1", delta=f"-{ratio_drop:.2f}")
        c2.metric("Lost 5-year CLV", f"${baseline_total - failure_total:,.2f}")
        c3.metric("Angry customers assigned", f"{len(assignments):,}")
        c4.metric("Requested angry share", f"{scenario_run['angry_percent']:.0f}%")
        year_counts = assignments["Failure year"].value_counts().reindex(
            [f"Year {year}" for year in YEARS], fill_value=0
        )
        st.caption("Random failure-year distribution: " + ", ".join(
            f"{year}: {count}" for year, count in year_counts.items()
        ))

        figure = comparison_figure(baseline_summary, failure_summary, run["firm"])
        st.pyplot(figure, use_container_width=True)
        image_buffer = BytesIO()
        figure.savefig(image_buffer, format="png", dpi=200, bbox_inches="tight")
        plt.close(figure)

        comparison = baseline_summary[["Year", "Returning profiles", "Annual CLV contribution"]].merge(
            failure_summary[["Year", "Returning profiles", "Annual CLV contribution"]], on="Year",
            suffixes=(" — baseline", " — service failure"),
        )
        comparison["Lost CLV"] = (
            comparison["Annual CLV contribution — baseline"]
            - comparison["Annual CLV contribution — service failure"]
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.subheader("Feedback: how to improve")
        feedback = assignments.copy()
        feedback["Recommended response"] = feedback["Dissatisfaction reason"].map(recovery_suggestion)
        st.dataframe(feedback, use_container_width=True, hide_index=True)
        st.success(
            "Priority: start with the year/reason producing the greatest downstream CLV loss, fix the root cause, "
            "and rerun the scenario with a lower angry-customer proportion to quantify the value of improvement."
        )

        stem = safe_filename(run["firm"])
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Download comparison", comparison.to_csv(index=False).encode("utf-8-sig"), f"{stem}_clv_comparison.csv", "text/csv")
        d2.download_button("Download affected customers", assignments.to_csv(index=False).encode("utf-8-sig"), f"{stem}_service_failures.csv", "text/csv")
        d3.download_button("Download feedback", feedback.to_csv(index=False).encode("utf-8-sig"), f"{stem}_recovery_feedback.csv", "text/csv")
        d4.download_button("Download chart", image_buffer.getvalue(), f"{stem}_service_failure_impact.png", "image/png")

    with st.expander("Baseline profile decisions and calculations"):
        st.dataframe(baseline_summary, use_container_width=True, hide_index=True)
        st.dataframe(results, use_container_width=True, height=420)
        failed = len(results) - len(successful)
        if failed:
            st.warning(f"{failed} failed simulation(s) were excluded from CLV calculations.")
