# demo_ui.py


import streamlit as st
from main import generate_transactions_from_text
import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
def generate_network_graph(entities, df):
    G = nx.DiGraph()
    # Add entity nodes
    for category, names in entities.get("Entities", {}).items():
        # Normalize type: e.g., "Individuals" -> "individual"
        node_type = category[:-1].lower() if category.endswith('s') else category.lower()
        for name in names:
            G.add_node(name, label=name, type=node_type)
    # Add account nodes 
    for acct in entities.get("Account_IDs", []):
        G.add_node(acct, label=acct, type="account")
        
    # Add account-to-FI edges
    for acct, fi in entities.get("Acct_to_FI", {}).items():
        if G.has_node(acct) and G.has_node(fi):
            G.add_edge(acct, fi, relation="held_at")
    # Add account-to-customer edges
    for acct, cust in entities.get("Acct_to_Cust", {}).items():
        if G.has_node(acct) and G.has_node(cust):
            G.add_edge(acct, cust, relation="owned_by")
    # Aggregate transactions by originator, beneficiary, and channel
    agg = df.groupby(
        ["Originator_Account_ID", "Beneficiary_Account_ID", "Trxn_Channel"],
        dropna=False
    )["Trxn_Amount"].sum().reset_index(name="total_amount")
    # Add one edge per channel with total amount
    for _, row in agg.iterrows():
        src = row["Originator_Account_ID"]
        dst = row["Beneficiary_Account_ID"]
        channel = row["Trxn_Channel"]
        if src and dst:
            edge_attrs = {
                "channel": channel,
                "type": channel,
                "total_amount": row["total_amount"]
            }
            # If Cash, capture all unique locations
            if channel == "Cash":
                locs = df.loc[
                    (df["Originator_Account_ID"] == src) &
                    (df["Beneficiary_Account_ID"] == dst) &
                    (df["Trxn_Channel"] == "Cash"),
                    "Branch_or_ATM_Location"
                ].dropna().unique().tolist()
                if locs:
                    edge_attrs["locations"] = locs
            G.add_edge(src, dst, **edge_attrs)
    # Record distinct transaction channels for downstream use
    channels = sorted(df["Trxn_Channel"].dropna().unique().tolist())
    G.graph["channels"] = channels
    return G



def render_legend(channel_colors, node_type_colors):
    """Render legends for channels, node types, and edge styles in Streamlit."""
    st.markdown("### 🔎 Legend")
    # Channels legend (dashed edges)
    with st.expander("Channels (dashed edges)", expanded=True):
        for ch, color in channel_colors.items():
            line_html = f"<span style='border-bottom:3px dotted {color}; display:inline-block; width:50px; margin-right:8px;'></span> {ch}"
            st.markdown(line_html, unsafe_allow_html=True)
    # Node types legend
    with st.expander("Node Types", expanded=True):
        for nt, color in node_type_colors.items():
            label = nt.replace('_', ' ').title()
            st.markdown(f"<span style='color:{color}'>■</span> {label}", unsafe_allow_html=True)
    # Edge styles legend
    with st.expander("Edge Styles", expanded=False):
        st.markdown("<span style='border-bottom:2px dashed black; display:inline-block; width:30px;'></span> Transaction Edge", unsafe_allow_html=True)
        st.markdown("<span style='border-bottom:2px solid black; display:inline-block; width:30px;'></span> Other Edge", unsafe_allow_html=True)


def visualize_interactive_graph(G):
    # Create a PyVis network
    # Define colors for each node type
    node_type_colors = {
        "account": "#1f77b4",
        "individual": "#ff7f0e",
        "organization": "#2ca02c",
        "financial_institution": "#d62728"
    }
    net = Network(height="600px", width="100%", directed=True)
    net.toggle_physics(True)
    # Add nodes
    for node, data in G.nodes(data=True):
        label = data.get("label", node)
        title = "<br>".join(f"{k}: {v}" for k, v in data.items())
        node_type = data.get("type", "")
        color = node_type_colors.get(node_type, "#7f7f7f")
        net.add_node(node, label=label, title=title, color=color)
    # Add edges with channel-specific colors and tooltips
    # Retrieve channels passed via the graph object
    channels = G.graph.get("channels", [])
    # Standard 8-color HTML palette
    palette = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f'   # middle gray
    ]
    channel_colors = {ch: palette[i % len(palette)] for i, ch in enumerate(channels)}
    for u, v, data in G.edges(data=True):
        channel = data.get("channel")
        if channel:
            color = channel_colors.get(channel, "black")
            dashed = True
        else:
            color = "black"
            dashed = False
        title = "<br>".join(f"{k}: {v}" for k, v in data.items())
        net.add_edge(u, v, title=title, color=color, arrows="to", dashes=dashed)
    # Generate and embed HTML
    html = net.generate_html(notebook=False)
    components.html(html, height=650)

    render_legend(channel_colors, node_type_colors)




st.set_page_config(page_title="SAR to Transactions", layout="wide")

st.title("🔍 SAR to Transactions")

uploaded_file = st.file_uploader("Upload a SAR file (.txt)", type="txt")

if uploaded_file is not None:
    sar_text = uploaded_file.read().decode("utf-8")

    st.subheader("📄 SAR Input Preview")
    st.text_area("SAR Content", sar_text, height=200)

    if st.button("Generate Transactions"):
        with st.spinner("Generating synthetic transactions..."):
            entities,df = generate_transactions_from_text(sar_text)
            # with open("./data/output/results_entity_metrics_20250615_111432.json", "r") as f:
            #     entities = json.load(f)
            # df = pd.read_csv("./data/output/results_trxns_20250615_111446.csv")

        st.success("Done!")

        st.subheader("📊 Generated Transactions")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "synthetic_transactions.csv", "text/csv")

        st.subheader("🌐 Transaction Graph")
        G = generate_network_graph(entities, df)
        visualize_interactive_graph(G)