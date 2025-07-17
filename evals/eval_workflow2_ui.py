import streamlit as st
import pandas as pd

@st.cache_data
def load_data(path: str = "./data/output/evals/workflow2/master_trxn_metrics.csv") -> pd.DataFrame:
    """
    Load the master_entity_metrics.csv file and parse the timestamp column.
    """
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

def main():
    st.title("SAR Metrics Trend Visualization")

    # Sidebar controls
    data_path = st.sidebar.text_input("CSV file path", "./data/output/evals/workflow2/master_trxn_metrics.csv")
    df = load_data(data_path)

    sar_options = sorted(df['sar_id'].unique().tolist())
    sar_choice = st.sidebar.selectbox("Select sar_id", sar_options)
    df_sar = df[df['sar_id'] == sar_choice]

    trxn_options = sorted(df_sar['Trxn_Set_ID'].unique().tolist())
    trxn_choice = st.sidebar.selectbox("Select Transaction Set ID", trxn_options)
    df_filtered = df_sar[df_sar['Trxn_Set_ID'] == trxn_choice].sort_values('timestamp')

    # Convert list-valued columns to their lengths
    for col in ["Missing_channels","Extra_channels","Missing_locations","Extra_locations"]:
        df_filtered[col] = df_filtered[col].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else x)

    # Convert boolean match fields to integers for plotting
    for bool_col in ['Channels_match', 'Location_Match']:
        if bool_col in df_filtered.columns:
            df_filtered[bool_col] = df_filtered[bool_col].astype(int)

    # Create a numeric index for unique timestamps
    unique_ts = sorted(df_filtered['timestamp'].unique())
    ts_to_idx = {ts: i+1 for i, ts in enumerate(unique_ts)}
    df_filtered['time_idx'] = df_filtered['timestamp'].map(ts_to_idx)

    # Metrics selection
    metric_columns = [
        col for col in df_filtered.columns 
        if col not in ['timestamp', 'sar_id', 'Trxn_Set_ID', 'time_idx']
    ]
    selected_metrics = st.sidebar.multiselect(
        "Select metrics to plot", 
        metric_columns, 
        default=metric_columns
    )

    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return

    # Prepare data for plotting
    plot_df = df_filtered[['time_idx'] + selected_metrics].set_index('time_idx')

    # Render line chart
    st.line_chart(plot_df)

if __name__ == "__main__":
    main()