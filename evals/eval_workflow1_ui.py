import streamlit as st
import pandas as pd

@st.cache_data
def load_data(path: str = "./data/output/evals/workflow1/master_entity_metrics.csv") -> pd.DataFrame:
    """
    Load the master_entity_metrics.csv file and parse the timestamp column.
    """
    df = pd.read_csv(path, parse_dates=['timestamp'])
    return df

def main():
    st.title("SAR Metrics Trend Visualization")

    # Sidebar controls
    data_path = st.sidebar.text_input("CSV file path", "./data/output/evals/workflow1/master_entity_metrics.csv")
    df = load_data(data_path)

    sar_options = sorted(df['SAR_index'].unique().tolist())
    sar_choice = st.sidebar.selectbox("Select SAR_index", sar_options)

    # Filter by SAR_index
    df_filtered = df[df['SAR_index'] == sar_choice].sort_values('timestamp')

    # Compute difference column
    df_filtered['diff_trxn_sets'] = (
        df_filtered['N_observed_trxn_sets'] - df_filtered['N_expected_trxn_sets']
    )

    # Create a numeric index for unique timestamps
    unique_ts = sorted(df_filtered['timestamp'].unique())
    ts_to_idx = {ts: i+1 for i, ts in enumerate(unique_ts)}
    df_filtered['time_idx'] = df_filtered['timestamp'].map(ts_to_idx)

    # Metrics selection
    metric_columns = [
        col for col in df_filtered.columns 
        if col not in ['timestamp', 'SAR_index']
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