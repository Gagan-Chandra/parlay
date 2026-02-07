import streamlit as st
import pandas as pd
import json
import requests
import os
from typing import Dict, Any, List


# -------------------------------
# 🧹 0. JSON parsing helper (robust)
# -------------------------------

def safe_parse_llm_json(raw: str) -> Dict[str, Any]:
    """
    Try very hard to turn the LLM response into a dict.
    Handles:
    - extra text before/after JSON
    - ```json ... ``` fences
    - models that return two JSON objects
    """
    text = raw.strip()

    # 1) direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) strip code fences
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except Exception:
            pass

    # 3) take largest {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 4) last resort
    return {
        "_parse_error": "Could not parse LLM JSON cleanly",
        "raw_text": raw[:2000],
    }


# -------------------------------
# 🔐 1. Call LLM Providers (Groq / Cerebras)
# -------------------------------

def call_llm_provider(
    provider: str,
    api_key: str,
    prompt: str,
    model: str = "",
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> str:
    """HTTP callers for Groq and Cerebras with full error details."""
    if provider == "Groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload)
        if not resp.ok:
            raise RuntimeError(f"Groq error {resp.status_code}: {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

    elif provider == "Cerebras":
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model or "llama3.1-70b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload)
        if not resp.ok:
            raise RuntimeError(f"Cerebras error {resp.status_code}: {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]

    else:
        raise ValueError("Unsupported provider. Use Groq or Cerebras.")


# -------------------------------
# 🧠 2. Deterministic rule helpers
# -------------------------------

PURPOSES = [
    "Working Capital",
    "Business Expansion",
    "Equipment Purchase or Leasing",
    "Inventory Purchase",
    "Real Estate Acquisition or Improvement",
    "Business Acquisition or Buyout",
    "Refinancing Existing Debt",
    "Emergency Funds",
    "Franchise Financing",
    "Contract Financing",
    "Licensing or Permits",
    "Line of Credit Establishment",
]


def get_float(row: Dict[str, Any], keys: List[str]) -> float | None:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            try:
                return float(row[k])
            except Exception:
                continue
    return None


def get_bool(row: Dict[str, Any], keys: List[str]) -> bool | None:
    for k in keys:
        if k in row:
            v = row[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return v != 0
            if isinstance(v, str):
                t = v.strip().lower()
                if t in ["true", "yes", "y", "1"]:
                    return True
                if t in ["false", "no", "n", "0"]:
                    return False
    return None


def extract_purposes(row: Dict[str, Any]) -> List[str]:
    """
    Detect which of the 12 PURPOSES are selected, from:
    - a single Purpose column
    - one-hot / boolean / 0-1 fields with purpose names / variants
    """
    selected = set()

    # 1) single 'Purpose' style field
    for k, v in row.items():
        if not isinstance(v, str):
            continue
        val = v.lower()
        for p in PURPOSES:
            if p.lower() in val:
                selected.add(p)

    # 2) flag-like columns
    for p in PURPOSES:
        p_slug = p.lower().replace(" ", "_")
        for k, v in row.items():
            kl = k.lower().replace(" ", "_")
            # column name mentions this purpose
            if p_slug in kl:
                if isinstance(v, bool) and v:
                    selected.add(p)
                elif isinstance(v, (int, float)) and v != 0:
                    selected.add(p)
                elif isinstance(v, str):
                    t = v.strip().lower()
                    if t in ["true", "yes", "y", "1"]:
                        selected.add(p)

    return list(selected)


def compute_loan_parameter_checks(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Apply your exact rules for 7(a), 504, Express and compute:
    - parameters_used
    - failed_rules
    - eligible
    """
    checks = []

    # Common extracted fields
    for_profit = get_bool(row, ["For Profit", "For_Profit", "for_profit"])
    pc = get_float(row, ["Personal Credit Score", "personal_credit_score", "Personal_Credit_Score"])
    bc = get_float(row, ["Business Credit Score", "business_credit_score", "Business_Credit_Score"])
    dscr = get_float(row, ["DSCR (latest year)", "dscr_latest", "DSCR_latest"])
    years = get_float(row, ["Years in Business", "years_in_business"])
    loan_amount = get_float(row, ["Loan Amount", "loan_amount"])
    collateral = get_bool(row, ["Collateral Availability", "collateral_available"])
    fast_approval = get_bool(row, ["Fast Approval", "fast_approval"])
    net_profit_margin = get_float(row, ["Net Profit Margin", "net_profit_margin"])
    purposes = extract_purposes(row)
    purpose_selected = purposes[0] if purposes else None

    has_valid_purpose = len(purposes) > 0

    # ---------- SBA 7(a) ----------
    failed_7a = []
    params_7a = {
        "for_profit": for_profit,
        "personal_credit_score": pc,
        "business_credit_score": bc,
        "dscr_latest": dscr,
        "years_in_business": years,
        "loan_amount": loan_amount,
        "purpose_selected": purpose_selected,
        "purpose_flags_raw": purposes,
    }

    if for_profit is not True:
        failed_7a.append("for_profit is False or missing for SBA 7(a)")
    if pc is None:
        failed_7a.append("missing Personal Credit Score for SBA 7(a), need >= 680")
    elif pc < 680:
        failed_7a.append(f"personal_credit_score {pc} < minimum 680 for SBA 7(a)")
    if bc is None:
        failed_7a.append("missing Business Credit Score for SBA 7(a), need >= 160")
    elif bc < 160:
        failed_7a.append(f"business_credit_score {bc} < minimum 160 for SBA 7(a)")
    if dscr is None:
        failed_7a.append("missing DSCR (latest year) for SBA 7(a), need >= 1.15")
    elif dscr < 1.15:
        failed_7a.append(f"dscr_latest {dscr} < minimum 1.15 for SBA 7(a)")
    if years is None:
        failed_7a.append("missing Years in Business for SBA 7(a), need >= 2")
    elif years < 2:
        failed_7a.append(f"years_in_business {years} < minimum 2 for SBA 7(a)")
    if loan_amount is None:
        failed_7a.append("missing Loan Amount for SBA 7(a), need between 500001 and 5000000")
    elif not (500001 <= loan_amount <= 5_000_000):
        failed_7a.append(f"loan_amount {loan_amount} not in [500001, 5000000] for SBA 7(a)")
    if not has_valid_purpose:
        failed_7a.append("no valid purpose selected for SBA 7(a)")
    if "Real Estate Acquisition or Improvement" in purposes:
        failed_7a.append("purpose 'Real Estate Acquisition or Improvement' must be False for SBA 7(a)")
    if "Emergency Funds" in purposes:
        failed_7a.append("purpose 'Emergency Funds' must be False for SBA 7(a)")

    eligible_7a = len(failed_7a) == 0
    checks.append(
        {
            "loan_type": "SBA 7(a)",
            "eligible": eligible_7a,
            "parameters_used": params_7a,
            "failed_rules": failed_7a,
        }
    )

    # ---------- SBA 504 ----------
    failed_504 = []
    params_504 = {
        "for_profit": for_profit,
        "collateral_available": collateral,
        "personal_credit_score": pc,
        "dscr_latest": dscr,
        "net_profit_margin": net_profit_margin,
        "years_in_business": years,
        "loan_amount": loan_amount,
        "purpose_selected": purpose_selected,
        "purpose_flags_raw": purposes,
    }

    if for_profit is not True:
        failed_504.append("for_profit is False or missing for SBA 504")
    if collateral is not True:
        failed_504.append("collateral_available is False or missing for SBA 504")
    if pc is None:
        failed_504.append("missing Personal Credit Score for SBA 504, need >= 680")
    elif pc < 680:
        failed_504.append(f"personal_credit_score {pc} < minimum 680 for SBA 504")
    if dscr is None:
        failed_504.append("missing DSCR (latest year) for SBA 504, need >= 1.15")
    elif dscr < 1.15:
        failed_504.append(f"dscr_latest {dscr} < minimum 1.15 for SBA 504")
    if net_profit_margin is None:
        failed_504.append("missing Net Profit Margin for SBA 504, need > 0")
    elif net_profit_margin <= 0:
        failed_504.append(f"net_profit_margin {net_profit_margin} <= 0 for SBA 504")
    if years is None:
        failed_504.append("missing Years in Business for SBA 504, need >= 2")
    elif years < 2:
        failed_504.append(f"years_in_business {years} < minimum 2 for SBA 504")
    if loan_amount is None:
        failed_504.append("missing Loan Amount for SBA 504, need <= 5500000")
    elif loan_amount > 5_500_000:
        failed_504.append(f"loan_amount {loan_amount} > maximum 5500000 for SBA 504")
    if not has_valid_purpose:
        failed_504.append("no valid purpose selected for SBA 504")
    for p in ["Working Capital", "Refinancing Existing Debt", "Emergency Funds"]:
        if p in purposes:
            failed_504.append(f"purpose '{p}' is not allowed for SBA 504")

    eligible_504 = len(failed_504) == 0
    checks.append(
        {
            "loan_type": "SBA 504",
            "eligible": eligible_504,
            "parameters_used": params_504,
            "failed_rules": failed_504,
        }
    )

    # ---------- SBA Express ----------
    failed_express = []
    params_express = {
        "for_profit": for_profit,
        "fast_approval": fast_approval,
        "personal_credit_score": pc,
        "business_credit_score": bc,
        "dscr_latest": dscr,
        "loan_amount": loan_amount,
        "purpose_selected": purpose_selected,
        "purpose_flags_raw": purposes,
    }

    if for_profit is not True:
        failed_express.append("for_profit is False or missing for SBA Express")
    if fast_approval is not True:
        failed_express.append("fast_approval is False or missing for SBA Express")
    if pc is None:
        failed_express.append("missing Personal Credit Score for SBA Express, need >= 680")
    elif pc < 680:
        failed_express.append(f"personal_credit_score {pc} < minimum 680 for SBA Express")
    if bc is None:
        failed_express.append("missing Business Credit Score for SBA Express, need >= 160")
    elif bc < 160:
        failed_express.append(f"business_credit_score {bc} < minimum 160 for SBA Express")
    if dscr is None:
        failed_express.append("missing DSCR (latest year) for SBA Express, need >= 1.15")
    elif dscr < 1.15:
        failed_express.append(f"dscr_latest {dscr} < minimum 1.15 for SBA Express")
    if loan_amount is None:
        failed_express.append("missing Loan Amount for SBA Express, need <= 500000")
    elif loan_amount > 500_000:
        failed_express.append(f"loan_amount {loan_amount} > maximum 500000 for SBA Express")
    if not has_valid_purpose:
        failed_express.append("no valid purpose selected for SBA Express")
    for p in ["Real Estate Acquisition or Improvement", "Business Acquisition or Buyout"]:
        if p in purposes:
            failed_express.append(f"purpose '{p}' is not allowed for SBA Express")

    eligible_express = len(failed_express) == 0
    checks.append(
        {
            "loan_type": "SBA Express",
            "eligible": eligible_express,
            "parameters_used": params_express,
            "failed_rules": failed_express,
        }
    )

    return checks


# -------------------------------
# 🧠 3. LLM rules summary (LLM uses audit, does NOT re-evaluate rules)
# -------------------------------

LOAN_RULES_SUMMARY = """
You are an AI Underwriter for small-business loans. Be strict, realistic, and auditable.

Use the following SBA references conceptually:
- SBA 7(a): https://www.sba.gov/funding-programs/loans/7a-loans
- SBA 504: https://www.sba.gov/funding-programs/loans/504-loans

IMPORTANT:
- The platform has ALREADY applied all numeric and eligibility rules
  for SBA 7(a), SBA 504, and SBA Express.
- You will be given a JSON array called loan_parameter_checks that shows,
  for each loan type:
  - whether it is eligible,
  - the parameters used,
  - which rules failed.

YOUR JOB:
- Do NOT re-calculate or override those rules.
- Instead, interpret them like a human underwriter:
  - pick the best program (or Ineligible),
  - explain why,
  - summarize strengths, weaknesses, and risk flags,
  - write an underwriter-style decision note.

You MUST:
- Use loan_parameter_checks as the ground truth for which programs pass/fail.
- If all three are ineligible → your final loan_type must be "Ineligible".
- If more than one is eligible, pick the most appropriate and explain why.
- Output ONLY JSON in the requested structure.
"""


# -------------------------------
# 🧱 4. Prompt builder – feeds in deterministic audit
# -------------------------------

def build_underwriter_prompt(applicant_dict: Dict[str, Any], loan_checks: list[Dict[str, Any]]) -> str:
    return f"""
{LOAN_RULES_SUMMARY}

Applicant data (raw row from platform):
{json.dumps(applicant_dict, indent=2)}

Deterministic rule audit from the platform (DO NOT change this data):
"loan_parameter_checks": {json.dumps(loan_checks, indent=2)}

Now return STRICT JSON in this EXACT structure:

{{
  "loan_recommendation": [
    {{
      "loan_type": "SBA 7(a) | SBA 504 | SBA Express | Ineligible",
      "confidence": "high | medium | low",
      "reason": "short paragraph referencing which programs passed/failed and key drivers (DSCR, credit, years, loan_amount, purpose, etc.)"
    }}
  ],
  "strengths": [
    "strength or positive signal 1",
    "strength or positive signal 2"
  ],
  "weaknesses": [
    "weakness or underwriting concern 1",
    "weakness or underwriting concern 2"
  ],
  "risk_flags": [
    "explicit risk 1",
    "explicit risk 2"
  ],
  "loan_parameter_checks": [
    "COPY EXACTLY the loan_parameter_checks JSON given above, without modification"
  ],
  "underwriter_note": "4–6 sentences written like a human underwriter, summarizing which programs were considered, which ones failed or passed, and why the final decision was made. Mention at least one of the SBA links where relevant."
}}

STRICT RULES FOR YOU:
- loan_parameter_checks in your output MUST be an exact copy of the platform audit provided (same three entries, same fields, same values).
- Do NOT edit, recompute, or 'fix' loan_parameter_checks. Just copy it.
- Base your recommendation, strengths, weaknesses, risk_flags, and underwriter_note on that audit.
- If all three loans are ineligible in the audit, your loan_recommendation[0].loan_type must be "Ineligible".
- Return ONLY JSON. No markdown, no comments, no extra text.
"""


# -------------------------------
# 📥 5. Load Uploaded File
# -------------------------------

@st.cache_data(show_spinner=False)
def load_applicants_from_file(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    return df


# -------------------------------
# 🖼️ 6. Streamlit Tab Renderer
# -------------------------------

def render_llm_underwriter_tab():
    st.subheader("🧠 LLM Underwriter (Groq / Cerebras)")

    col_p1, col_p2, col_p3 = st.columns([1, 1, 1.2])
    with col_p1:
        provider = st.selectbox("LLM Provider", ["Groq", "Cerebras"], index=0)
    with col_p2:
        if provider == "Groq":
            model_name = st.selectbox(
                "Groq Model",
                [
                    "llama-3.1-8b-instant",
                    "llama-3.1-70b-versatile",
                    "mixtral-8x7b-32768",
                ],
                index=0,
            )
        else:
            model_name = st.text_input("Cerebras Model", value="llama3.1-70b")

    provider_key_env = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(provider_key_env, "")

    with col_p3:
        if api_key:
            st.success(f"🔐 Using {provider_key_env} from environment")
        else:
            api_key = st.text_input("API Key", type="password")

    st.markdown("—")

    # file upload
    uploaded_file = st.file_uploader(
        "Upload applicant data (CSV/Excel with Applicant ID column)",
        type=["csv", "xlsx", "xls"],
    )
    df = load_applicants_from_file(uploaded_file)
    if df.empty:
        st.info("Upload data to start underwriting.")
        return

    # pick ID col
    id_col_candidates = [c for c in df.columns if c.lower() in ["applicant id", "applicant_id", "id", "loan_id"]]
    id_col = id_col_candidates[0] if id_col_candidates else st.selectbox("Select Applicant ID column", df.columns)
    applicant_ids = df[id_col].unique().tolist()
    selected_applicant_id = st.selectbox("Select Applicant", applicant_ids)

    applicant_row = df[df[id_col] == selected_applicant_id].iloc[0].to_dict()

    st.markdown("### 📄 Applicant Snapshot")
    st.json(applicant_row)
    st.markdown("—")

    temp = st.slider("Creativity / Temperature", 0.0, 1.0, 0.2)
    max_tok = st.slider("Max Tokens", 400, 1500, 900)
    run_btn = st.button("🔎 Run LLM Underwriting", use_container_width=True)

    if "llm_underwriter_runs" not in st.session_state:
        st.session_state["llm_underwriter_runs"] = []

    if run_btn:
        if not api_key:
            st.error("Please provide an API key (via .env or manual entry).")
        else:
            with st.spinner("Running LLM Underwriter..."):
                try:
                    # 1) deterministic audit
                    loan_checks = compute_loan_parameter_checks(applicant_row)

                    # 2) build prompt using that audit
                    prompt = build_underwriter_prompt(applicant_row, loan_checks)

                    # 3) LLM call
                    raw = call_llm_provider(
                        provider=provider,
                        api_key=api_key,
                        prompt=prompt,
                        model=model_name,
                        temperature=temp,
                        max_tokens=max_tok,
                    )
                    parsed = safe_parse_llm_json(raw)

                    # 4) regardless of what LLM did, we TRUST our loan_checks
                    parsed["loan_parameter_checks"] = loan_checks

                    st.session_state["llm_underwriter_runs"].insert(
                        0,
                        {
                            "applicant_id": selected_applicant_id,
                            "raw": raw,
                            "parsed": parsed,
                            "provider": provider,
                        },
                    )
                    st.success("✅ Underwriting complete.")
                except Exception as e:
                    st.error(f"LLM call failed: {e}")

    # --- show latest result ---
    if st.session_state["llm_underwriter_runs"]:
        latest = st.session_state["llm_underwriter_runs"][0]
        parsed = latest.get("parsed", {})

        # if parsing failed badly, show raw
        if parsed.get("_parse_error"):
            st.warning(parsed["_parse_error"])
            st.code(parsed.get("raw_text", latest.get("raw", "")))
            return

        st.markdown("### 📌 Loan Recommendation (LLM)")
        for rec in parsed.get("loan_recommendation", []):
            st.write(f"**{rec.get('loan_type')}** — confidence `{rec.get('confidence')}`")
            st.write(rec.get("reason", ""))

        st.markdown("#### ✅ Strengths (LLM)")
        for s in parsed.get("strengths", []):
            st.write(f"- {s}")

        st.markdown("#### ⚠️ Weaknesses (LLM)")
        for w in parsed.get("weaknesses", []):
            st.write(f"- {w}")

        st.markdown("#### 🚩 Risk Flags (LLM)")
        for r in parsed.get("risk_flags", []):
            st.write(f"- {r}")

        st.markdown("#### 🧪 Loan Parameter Checks (Platform Rules)")
        for check in parsed.get("loan_parameter_checks", []):
            st.write(f"**{check.get('loan_type','(unknown loan)')}** — eligible: `{check.get('eligible', False)}`")
            st.write("Parameters used:")
            st.json(check.get("parameters_used", {}))
            failed_rules = check.get("failed_rules", [])
            if failed_rules:
                st.write("Failed rules:")
                for fr in failed_rules:
                    st.write(f"- {fr}")
            else:
                st.write("Failed rules: [] (all rules passed)")
            st.write("---")

        st.markdown("#### 📝 Underwriter Note (LLM)")
        st.write(parsed.get("underwriter_note", ""))

        with st.expander("🔍 Raw JSON Output from LLM (after merging audit)"):
            st.json(parsed)
    else:
        st.info("Run the model to see results.")


# -------------------------------
# 🔁 Entry point for app.py
# -------------------------------

def render():
    render_llm_underwriter_tab()
