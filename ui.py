# demo_ui.py


import streamlit as st
from main import generate_transactions_from_text
import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
def generate_network_graph(entities, df):
    G = nx.DiGraph()
    # Add entity nodes
    for category, names in entities.get("Entities", {}).items():
        # Normalize type: e.g., "Individuals" -> "individual"
        node_type = category[:-1].lower() if category.endswith('s') else category.lower()
        for name in names:
            G.add_node(name, label=name, type=node_type)
    # Add account nodes and narratives
    for acct in entities.get("Account_IDs", []):
        G.add_node(acct, label=acct, type="account")
        narrative = entities.get("Narratives", {}).get(acct)
        if narrative:
            G.nodes[acct]["narrative"] = narrative
    # Add account-to-FI edges
    for acct, fi in entities.get("Acct_to_FI", {}).items():
        if G.has_node(acct) and G.has_node(fi):
            G.add_edge(acct, fi, relation="held_at")
    # Add account-to-customer edges
    for acct, cust in entities.get("Acct_to_Cust", {}).items():
        if G.has_node(acct) and G.has_node(cust):
            G.add_edge(acct, cust, relation="owned_by")
    # Add transaction edges
    for _, row in df.iterrows():
        src = row.get("Originator_Account_ID")
        dst = row.get("Beneficiary_Account_ID")
        if src and dst:
            G.add_edge(
                src,
                dst,
                channel=row.get("Trxn_Channel"),
                date=row.get("Trxn_Date"),
                amount=row.get("Trxn_Amount"),
                location=row.get("Branch_or_ATM_Location"),
                transaction_id=row.get("Transaction_ID")
            )
    return G

def visualize_graph(G):
    # Use spring layout for positioning
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    # Draw nodes
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels)
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True)
    # Optionally, draw edge labels for relations or transaction IDs
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        # show relation if present, else transaction_id
        lbl = data.get('relation') or data.get('transaction_id', '')
        if lbl:
            edge_labels[(u, v)] = lbl
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis('off')
    st.pyplot(plt)





st.set_page_config(page_title="SAR to Transactions Demo", layout="wide")

st.title("🔍 SAR to Synthetic Transactions Demo")

uploaded_file = st.file_uploader("Upload a SAR file (.txt)", type="txt")

if uploaded_file is not None:
    sar_text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 SAR Input Preview")
    st.text_area("SAR Content", sar_text, height=200)

    if st.button("Generate Transactions"):
        with st.spinner("Generating synthetic transactions..."):
            #entities,df = generate_transactions_from_text(sar_text)
            with open("./data/output/results_entity_metrics_20250615_111432.json", "r") as f:
                entities = json.load(f)
            df = pd.read_csv("./data/output/results_trxns_20250615_111446.csv")

        st.success("Done!")

        st.subheader("📊 Generated Transactions")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "synthetic_transactions.csv", "text/csv")

        st.subheader("🌐 Transaction Graph")
        G = generate_network_graph(entities, df)
        visualize_graph(G)