import io
import re
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import pdfplumber
from rapidfuzz import process, fuzz

# ----------------- OpenAI GPT Setup -----------------
import openai

OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # <--- Replace with your OpenAI API key
openai.api_key = OPENAI_API_KEY
OPENAI_MODEL = "gpt-4o-mini"

# Optional OCR imports (handled gracefully if missing)
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# --------- Streamlit page config and styles ---------
st.set_page_config(page_title="AI Personal Finance Assistant", page_icon="💰", layout="wide")
st.markdown(
    """
    <style>
    .main-title { text-align:center; font-size:34px; font-weight:700; color:#4CAF50; text-shadow: 2px 2px 5px rgba(76,175,80,.4);} 
    .sub-title { text-align:center; font-size:18px; color:#ddd; margin-bottom:20px;}
    .result-card { background: rgba(0, 150, 136, 0.08); padding: 16px; border-radius: 10px; margin: 8px 0; box-shadow: 0 2px 8px rgba(0,0,0,.06);} 
    .success-banner { background: linear-gradient(to right, #2E7D32, #1B5E20); color:#fff; padding:14px; border-radius:10px; text-align:center; font-weight:700; margin-top:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<h1 class="main-title">💰 AI-Powered Personal Finance Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload your UPI/Bank statements (Paytm, GPay, PhonePe, etc.) → Parse → Analyze → Get insights</p>', unsafe_allow_html=True)

# --------- Sidebar ---------
st.sidebar.title("ℹ️ How to Use")
st.sidebar.write("1) Upload one or more statement PDFs.")
st.sidebar.write("2) Review parsed transactions.")
st.sidebar.write("3) (Optional) Enter monthly income for savings%.")
st.sidebar.write("4) Generate AI insights or use offline analytics.")

with st.sidebar.expander("⚙️ Settings", expanded=False):
    user_monthly_income = st.number_input("Monthly Income (₹) — optional", min_value=0, step=1000, value=0)
    enable_ocr = st.checkbox("Enable OCR fallback (slower)", value=False)
    chunk_limit = st.number_input("Max rows sent to LLM", min_value=100, max_value=5000, value=800, step=100)
    show_offline_insights = st.checkbox("Show offline insights (no LLM)", value=True)

# --------- Helpers (parsing, normalization, categorization) ---------
DATE_PATTERNS = ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y", "%d %B %Y"]
DATE_RE = re.compile(r"\b(\d{2}[/-]\d{2}[/-]\d{4}|\d{4}-\d{2}-\d{2}|\d{2}\s?[A-Za-z]{3}\s?\d{4})\b")
AMOUNT_RE = re.compile(r"(?:₹|INR)?\s*([+-]?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})|[+-]?\d+(?:\.\d{1,2}))")
DEBIT_KEYS = {"debit", "debited", "withdrawal", "withdraw", "purchase", "payment", "paid", "upi to",
              "money sent", "sent", "transfer to", "atm", "pos", "bill", "emi", "petrol", "fuel", "dining",
              "zomato", "swiggy", "insurance premium", "online purchase"}
CREDIT_KEYS = {"credit", "credited", "cr", "received", "refund", "reversal", "cashback", "salary",
               "deposit", "transfer from", "interest", "redeem", "redemption"}
BALANCE_KEYS = {"opening balance", "closing balance", "balance only", "closingbal", "openingbal"}
CATEGORY_RULES = {
    "food": ["swiggy", "zomato", "restaurant", "hotel", "food"],
    "grocery": ["bigbasket", "dmart", "grocery", "mart", "supermarket"],
    "utilities": ["electricity", "water", "gas", "biller", "broadband", "internet", "dth"],
    "transport": ["uber", "ola", "metro", "fuel", "petrol", "diesel", "fastag"],
    "shopping": ["amazon", "flipkart", "myntra", "ajio", "store", "shop"],
    "entertainment": ["netflix", "prime", "hotstar", "movie", "bookmyshow"],
    "financial": ["bank", "emi", "loan", "insurance", "sip", "mutual fund"],
    "transfer": ["upi", "imps", "neft", "rtgs", "to", "from"],
}
UPI_ID_RE = re.compile(r"[A-Za-z0-9._-]+@[A-Za-z]+")
MERCHANT_CANON = {"swiggy": "Swiggy", "zomato": "Zomato", "amazon": "Amazon", "flipkart": "Flipkart",
                  "uber": "Uber", "ola": "Ola"}

def _normalize_date(s: str) -> str | None:
    s = s.strip()
    for fmt in DATE_PATTERNS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None

def _extract_texts_from_pdf(file_bytes: bytes, use_ocr: bool = False) -> str:
    text_chunks: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if (not t) and use_ocr and OCR_AVAILABLE:
                try:
                    images = convert_from_bytes(file_bytes, first_page=page.page_number, last_page=page.page_number)
                    if images:
                        t = pytesseract.image_to_string(images[0]) or ""
                except Exception:
                    t = ""
            text_chunks.append(t)
    return "\n".join(text_chunks).strip()

def _canonical_merchant(name: str) -> str:
    low = (name or "").lower()
    for key, canon in MERCHANT_CANON.items():
        if key in low:
            return canon
    try:
        best = process.extractOne(low, list(MERCHANT_CANON.keys()), scorer=fuzz.WRatio)
        if best and isinstance(best, tuple) and len(best) >= 2 and best[1] > 85:
            return MERCHANT_CANON.get(best[0], name)
    except Exception:
        pass
    return (name or "").strip()[:80]

def _categorize(desc: str) -> str:
    d = (desc or "").lower()
    for cat, kws in CATEGORY_RULES.items():
        if any(k in d for k in kws):
            return cat
    return "other"

def _find_amounts(txt: str) -> list[float]:
    vals: list[float] = []
    for m in AMOUNT_RE.findall(txt or ""):
        s = m if isinstance(m, str) else (m[0] if isinstance(m, (list, tuple)) else str(m))
        try:
            vals.append(float(s.replace(",", "")))
        except Exception:
            pass
    return vals

def _guess_type_from_text(txt: str) -> str:
    low = (txt or "").lower()
    if any(k in low for k in BALANCE_KEYS):
        return "balance"
    if re.search(r"\bcr\b", low):
        return "credit"
    if re.search(r"\bdr\b", low):
        return "debit"
    if any(k in low for k in CREDIT_KEYS):
        return "credit"
    if any(k in low for k in DEBIT_KEYS):
        return "debit"
    if "transfer from" in low or ("from" in low and "transfer" in low):
        return "credit"
    if "transfer to" in low or ("to" in low and "transfer" in low):
        return "debit"
    return "unknown"

def _extract_amount_and_type(txt: str) -> tuple[float | None, str, float | None]:
    amounts = _find_amounts(txt)
    tx_type = _guess_type_from_text(txt)
    if tx_type == "balance" and len(amounts) == 1:
        return (None, "balance_only", amounts[0])
    balance_amt = None
    tx_amt = None
    if len(amounts) >= 2:
        balance_amt = amounts[-1]
        tx_amt = amounts[0]
    elif len(amounts) == 1:
        tx_amt = amounts[0]
    return (tx_amt, tx_type, balance_amt)

# ----------------------- Parsing pipeline -----------------------
@st.cache_data(show_spinner=False)
def extract_and_parse(files: List[Tuple[str, bytes]], use_ocr: bool = False) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []
    for filename, file_bytes in files:
        raw_text = _extract_texts_from_pdf(file_bytes, use_ocr=use_ocr)
        source = filename
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        block: List[str] = []
        for ln in lines:
            if DATE_RE.search(ln) and block:
                all_rows.extend(_parse_block(" ".join(block), source))
                block = [ln]
            else:
                block.append(ln)
        if block:
            all_rows.extend(_parse_block(" ".join(block), source))
    df = pd.DataFrame(all_rows).drop_duplicates()
    if not df.empty:
        df["date"] = df["date"].apply(lambda x: _normalize_date(str(x)) or "")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.dropna(subset=["amount"]).reset_index(drop=True)
        with pd.option_context('mode.chained_assignment', None):
            df["month"] = df["date"].apply(lambda s: s[:7] if isinstance(s, str) and len(s) >= 7 else "")
        df["merchant"] = df["merchant"].fillna("").apply(_canonical_merchant)
        df["category"] = df["description"].fillna("").apply(_categorize)
        df["type"] = df["type"].fillna("")
        df.loc[~df["type"].isin(["debit", "credit"]), "type"] = "unknown"
        if "currency" not in df.columns:
            df["currency"] = "INR"
        if "transaction_id" not in df.columns:
            df["transaction_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        df["signed_amount"] = np.where(
            df["type"] == "credit", df["amount"],
            np.where(df["type"] == "debit", -df["amount"], np.nan)
        )
    return df

def _parse_block(txt: str, source_pdf: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    date_match = DATE_RE.search(txt or "")
    date_str = date_match.group(0) if date_match else ""
    tx_amt, tx_type, balance_amt = _extract_amount_and_type(txt or "")
    if tx_type in {"balance_only", "balance"} and tx_amt is None:
        return rows
    upi_match = UPI_ID_RE.search(txt or "")
    merchant = upi_match.group(0) if upi_match else ""
    if not merchant:
        tokens = [t for t in re.split(r"[^A-Za-z0-9@._-]+", txt or "") if t and not t.replace('.', '', 1).isdigit()]
        merchant = max(tokens, key=len)[:80] if tokens else ""
    if tx_type == "unknown":
        low = (txt or "").lower()
        if any(k in low for k in ["salary", "deposit", "refund", "received", "cashback"]):
            tx_type = "credit"
        elif any(k in low for k in ["withdraw", "payment", "purchase", "upi to", "sent", " to "]):
            tx_type = "debit"
    if tx_amt is not None:
        rows.append({
            "date": date_str,
            "time": "",
            "amount": float(tx_amt),
            "type": tx_type if tx_type in {"debit", "credit"} else "unknown",
            "merchant": merchant,
            "description": (txt or "").strip()[:300],
            "method": _infer_method_from_source(source_pdf),
            "source_pdf": source_pdf,
            "running_balance": balance_amt if balance_amt is not None else np.nan,
        })
    return rows

def _infer_method_from_source(name: str) -> str:
    low = (name or "").lower()
    if "paytm" in low:
        return "Paytm"
    if "phonepe" in low:
        return "PhonePe"
    if "gpay" in low or "google" in low:
        return "GPay"
    return "UPI"

# ----------------------- Offline analytics -----------------------
def compute_offline_insights(df: pd.DataFrame, income: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {"months": {}, "top_categories": [], "top_merchants": [], "notes": []}
    if df.empty:
        return out
    month_grp = df[df["type"].isin(["debit", "credit"])].groupby(["month", "type"]).agg(total=("amount", "sum")).reset_index()
    months: Dict[str, Dict[str, float]] = {}
    for _, r in month_grp.iterrows():
        m = r["month"] or "unknown"
        if m not in months:
            months[m] = {"income": 0.0, "expenses": 0.0}
        if r["type"] == "credit":
            months[m]["income"] += float(r["total"])
        elif r["type"] == "debit":
            months[m]["expenses"] += float(r["total"])
    out["months"] = months
    cat = df[df["type"] == "debit"].groupby("category")["amount"].sum().sort_values(ascending=False)
    out["top_categories"] = [(k, float(v)) for k, v in cat.head(10).items()]
    mer = df[df["type"] == "debit"].groupby("merchant")["amount"].sum().sort_values(ascending=False)
    out["top_merchants"] = [(k, float(v)) for k, v in mer.head(10).items()]
    notes: List[str] = []
    for m, vals in months.items():
        inc = income if income > 0 else vals.get("income", 0.0)
        exp = vals.get("expenses", 0.0)
        if inc > 0:
            savings_pct = max(0.0, (inc - exp) / inc * 100.0)
            notes.append(f"{m}: Savings {savings_pct:.1f}% (Income ₹{inc:,.0f}, Expenses ₹{exp:,.0f})")
        else:
            notes.append(f"{m}: Expenses ₹{exp:,.0f} (No income provided/detected)")
    out["notes"] = notes
    return out

# ----------------------- LLM GPT Context -----------------------
def build_llm_context(df: pd.DataFrame, max_rows: int = 800) -> str:
    cols = ["date", "amount", "type", "merchant", "category", "method"]
    slim = df[cols].copy() if not df.empty else pd.DataFrame(columns=cols)
    if len(slim) > max_rows:
        slim = slim.sort_values("date").tail(max_rows)
    return slim.to_json(orient="records", date_format="iso")

@st.cache_data(show_spinner=False)
def generate_recommendations_gpt(df: pd.DataFrame, income: int = 0, max_rows: int = 800) -> str:
    if df.empty:
        return "⚠️ No transactions to analyze."

    context_json = build_llm_context(df, max_rows=max_rows)

    system_msg = (
        "You are a practical, precise personal finance advisor. "
        "Use the provided JSON transactions to compute figures."
    )

    user_instructions = {
        "goals": [
            "1) Provide a monthly budget plan (monthly target per category and total monthly budget).",
            "2) Suggest concrete ways to reduce unnecessary or recurring spending with estimated ₹ savings.",
            "3) Provide 3 personalized financial recommendations (e.g., emergency fund, SIPs, debt paydown), with estimated impact in ₹ or % where possible."
        ],
        "notes": "Use transaction data strictly from DATA(JSON). If income is provided, use it for savings calculations; otherwise infer income from credits.",
        "output_format": {
            "sections": ["Monthly Budget Plan", "Reduce Spending Suggestions", "Personalized Advice"],
            "format": "Markdown with bullet lists and numeric estimates. Keep it concise."
        },
        "user_income": income
    }

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"DATA(JSON):\n{context_json}\n\nINSTRUCTIONS:\n{json.dumps(user_instructions, ensure_ascii=False)}"}
    ]

    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=1000
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM error: {str(e)}"

# ----------------------- Streamlit UI -----------------------
uploaded_files = st.file_uploader(
    "📂 Upload one or more PDF statements",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    files_payload = [(f.name, f.read()) for f in uploaded_files]
    st.success(f"✅ {len(files_payload)} file(s) uploaded.")

    if enable_ocr and not OCR_AVAILABLE:
        st.warning("OCR requested but OCR libraries (pytesseract/pdf2image) are not available. Parsing will proceed without OCR.")

    with st.spinner("📄 Extracting & parsing transactions..."):
        df = extract_and_parse(files_payload, use_ocr=(enable_ocr and OCR_AVAILABLE))

    if df.empty:
        st.error("No transactions parsed. Try enabling OCR or verify the PDF format.")
    else:
        st.subheader("📋 Parsed Transactions (preview)")
        st.dataframe(df.head(500), use_container_width=True)

        # Basic KPIs
        df_known = df[df["type"].isin(["debit", "credit"])].copy()
        total_exp = float(df_known.loc[df_known["type"] == "debit", "amount"].sum())
        total_inc = float(df_known.loc[df_known["type"] == "credit", "amount"].sum())
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Expenses (₹)", f"{total_exp:,.0f}")
        kpi2.metric("Total Income (₹)", f"{total_inc:,.0f}")
        net = total_inc - total_exp
        kpi3.metric("Net (₹)", f"{net:,.0f}", delta=None)

        # Charts & downloads
        tab1, tab2, tab3 = st.tabs(["📆 Monthly", "🗂️ Categories", "🏷️ Merchants"])
        with tab1:
            if "month" in df_known.columns and df_known["month"].nunique() > 0:
                msum = df_known.groupby(["month", "type"]).amount.sum().unstack(fill_value=0)
                st.bar_chart(msum)
        with tab2:
            cat_sum = df_known[df_known["type"] == "debit"].groupby("category").amount.sum().sort_values(ascending=False)
            st.bar_chart(cat_sum)
        with tab3:
            mer_sum = df_known[df_known["type"] == "debit"].groupby("merchant").amount.sum().sort_values(ascending=False).head(20)
            st.bar_chart(mer_sum)

        st.download_button(
            label="⬇️ Download Parsed CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="transactions_parsed.csv",
            mime="text/csv",
        )

        # Offline insights
        if show_offline_insights:
            st.subheader("📊 Offline Insights (no LLM)")
            off = compute_offline_insights(df, income=user_monthly_income)
            st.json(off)

        # LLM recommendations
        st.subheader("🧠 AI Recommendations")
        if st.button("Generate Recommendations (GPT-4o-mini)"):
            with st.spinner("Generating recommendations via GPT..."):
                md = generate_recommendations_gpt(df, income=user_monthly_income, max_rows=int(chunk_limit))
            st.markdown(md or "⚠️ No insights returned.")
            st.download_button(
                label="⬇️ Download Recommendations (Markdown)",
                data=(md or "").encode("utf-8"),
                file_name="financial_recommendations.md",
                mime="text/markdown",
            )
            st.markdown('<div class="success-banner">🎉 Recommendations ready — use them to plan your finances!</div>', unsafe_allow_html=True)
            try:
                st.balloons()
            except Exception:
                pass
else:
    st.info("Upload at least one PDF to begin.")