import io
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Import Layer 5 Orchestrator Engine
from scripts.orchestrator import run_customs_orchestrator


def generate_pdf_report(user_query: str, res: dict) -> bytes:
    """Generates an in-memory PDF assessment report using ReportLab."""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "PAKISTAN CUSTOMS DUTY ASSESSMENT REPORT")
    p.setStrokeColor(colors.gray)
    p.line(50, height - 60, width - 50, height - 60)

    # Input Summary
    p.setFont("Helvetica-Bold", 11)
    p.drawString(50, height - 85, "Query & Import Details:")
    p.setFont("Helvetica", 10)
    p.drawString(60, height - 100, f"User Input: {user_query}")

    duty_data = res.get("duty_calculation", {}) or res.get("calculation", {})
    inputs = duty_data.get("inputs", {})
    hs_code = res.get("hs_code", res.get("resolved_hs", "N/A"))
    fob = inputs.get("fob_usd", 0.0)
    ex_rate = inputs.get("exchange_rate", 0.0)
    cif_pkr = inputs.get("cif_pkr", 0.0)

    p.drawString(60, height - 115, f"Resolved HS Code: {hs_code}")
    p.drawString(60, height - 130, f"FOB (USD): ${fob:,.2f} | Exchange Rate: {ex_rate} PKR/USD")
    p.drawString(60, height - 145, f"Assessed Value (CIF PKR): Rs. {cif_pkr:,.2f}")

    # Waterfall Tax Table
    p.setFont("Helvetica-Bold", 11)
    p.drawString(50, height - 175, "Itemized Duty Waterfall (PKR):")

    duties = duty_data.get("duties_breakdown_pkr", {})
    y = height - 195

    p.setFont("Helvetica", 10)
    for tax_name, amount in duties.items():
        formatted_name = tax_name.replace("_", " ").title()
        p.drawString(70, y, f"• {formatted_name}:")
        p.drawRightString(width - 70, y, f"Rs. {amount:,.2f}")
        y -= 18

    # Footer
    p.line(50, y - 10, width - 50, y - 10)
    p.setFont("Helvetica-Oblique", 8)
    p.drawString(50, y - 25, "Generated automatically via Pakistan Customs AI Engine.")

    p.showPage()
    p.save()

    buffer.seek(0)
    return buffer.getvalue()


# ---------------- STREAMLIT UI LAYOUT ----------------

st.set_page_config(
    page_title="Pakistan Customs AI Duty Calculator",
    page_icon="🛃",
    layout="wide"
)

st.title("🛃 Pakistan Customs AI Engine & Duty Waterfall")
st.caption("AI-powered tariff resolution, legal context retrieval (RAG), and compounding duty calculation.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Import Financial Parameters")
    fob_usd = st.number_input("FOB Value (USD)", min_value=1.0, value=10000.0, step=100.0)
    freight_usd = st.number_input("Freight (USD)", min_value=0.0, value=100.0, step=10.0)
    insurance_usd = st.number_input("Insurance (USD)", min_value=0.0, value=50.0, step=10.0)

    st.divider()
    st.subheader("Tax Overrides (%)")
    ait_rate = st.number_input("Advance Income Tax - AIT (%)", min_value=0.0, max_value=30.0, value=6.0, step=0.5)
    fed_rate = st.number_input("Federal Excise Duty - FED (%)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)

# Main Query Inputs
col_q, col_hs = st.columns([3, 1])

with col_q:
    user_query = st.text_input("Describe the imported goods:", value="I want to import a mini van vehicle")

with col_hs:
    hs_code_override = st.text_input("Explicit HS Code (Optional):", value="")

# Calculate Button
if st.button("🚀 Calculate Customs Duty", type="primary", use_container_width=True):
    with st.spinner("Processing tariff resolution, currency rates, legal RAG, and duty waterfall..."):
        # Execute pipeline
        result = run_customs_orchestrator(
            user_query=user_query,
            fob_usd=fob_usd,
            hs_code=hs_code_override.strip() if hs_code_override.strip() else "",
            item_description=user_query,
            ait_rate=ait_rate,
            fed_rate=fed_rate,
            freight_usd=freight_usd,
            insurance_usd=insurance_usd
        )

    # Check for resolution errors
    if result.get("error"):
        st.error(f"❌ Resolution Error: {result.get('message', 'An error occurred during calculation.')}")
    else:
        # 1. Cache Indicator Banner
        cache_hit = result.get("cache_hit", False)
        cache_type = result.get("cache_type", "NONE")

        if cache_hit:
            st.success(f"⚡ **Instant Cache Hit!** Served via `{cache_type}` Cache in ~0.001s.")
        else:
            st.info("🤖 **Fresh AI Synthesis:** Resolved live via Gemini, ChromaDB RAG, and WeBOC waterfall calculations.")

        # Top KPI Summary Metrics
        duty_data = result.get("duty_calculation", {}) or result.get("calculation", {})
        breakdown = duty_data.get("duties_breakdown_pkr", {})
        inputs = duty_data.get("inputs", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Resolved HS Code", result.get("hs_code", result.get("resolved_hs", "N/A")))
        m2.metric("Exchange Rate", f"{inputs.get('exchange_rate', 0.0)} PKR/USD")
        m3.metric("Assessed Value (CIF)", f"Rs. {inputs.get('cif_pkr', 0.0):,.0f}")
        m4.metric("Total Tax Payable", f"Rs. {breakdown.get('total_payable', 0.0):,.0f}")

        st.divider()

        # 2. Multi-Tab Results Section
        tab_waterfall, tab_ai_report, tab_legal_rag = st.tabs([
            "📊 Duty Waterfall Breakdown", 
            "🤖 AI Summary Report", 
            "📜 Legal Context (RAG)"
        ])

        with tab_waterfall:
            st.subheader("Compounding Duty Waterfall Calculation")

            # Display applied rates
            rates = duty_data.get("rates_applied", {})
            st.markdown(
                f"**Applied Rates:** CD: `{rates.get('cd_rate', 0.0)}%` | "
                f"Sales Tax: `{rates.get('sales_tax_rate', 18.0)}%` | "
                f"AIT: `{ait_rate}%` | FED: `{fed_rate}%` | "
                f"Sindh Cess: `{rates.get('sindh_cess_rate', 0.0)}%`"
            )

            # Formatted Data Table
            table_rows = []
            for tax, val in breakdown.items():
                table_rows.append({
                    "Tax Head": tax.replace("_", " ").title(),
                    "Amount (PKR)": f"Rs. {val:,.2f}"
                })

            st.table(table_rows)

        with tab_ai_report:
            st.subheader("Gemini AI Executive Summary")
            st.markdown(result.get("summary_report", result.get("ai_report", "No summary generated.")))

        with tab_legal_rag:
            st.subheader("Retrieved Legal Context & SRO Snippets")
            legal_snippets = result.get("legal_snippets", [])

            if legal_snippets:
                for idx, snippet in enumerate(legal_snippets, 1):
                    source = snippet.get("source", "Legal Document") if isinstance(snippet, dict) else "Legal Document"
                    text = snippet.get("text", snippet) if isinstance(snippet, dict) else str(snippet)
                    with st.expander(f"Legal Excerpt #{idx} [{source}]"):
                        st.write(text)
            else:
                st.warning("No specific legal excerpts retrieved for this query.")

        # 3. PDF Export Download Button
        st.divider()
        pdf_data = generate_pdf_report(user_query, result)
        st.download_button(
            label="📄 Download Official PDF Duty Assessment",
            data=pdf_data,
            file_name=f"Customs_Assessment_{result.get('hs_code', 'Report')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )