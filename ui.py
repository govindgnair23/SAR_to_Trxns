# demo_ui.py

import streamlit as st
from main import generate_transactions_from_text

st.set_page_config(page_title="SAR to Transactions Demo", layout="wide")

st.title("🔍 SAR to Synthetic Transactions Demo")

uploaded_file = st.file_uploader("Upload a SAR file (.txt)", type="txt")

if uploaded_file is not None:
    sar_text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 SAR Input Preview")
    st.text_area("SAR Content", sar_text, height=200)

    if st.button("Generate Transactions"):
        with st.spinner("Generating synthetic transactions..."):
            df = generate_transactions_from_text(sar_text)

        st.success("Done!")

        st.subheader("📊 Generated Transactions")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "synthetic_transactions.csv", "text/csv")