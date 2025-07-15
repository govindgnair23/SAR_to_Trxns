
from evals.golden_data import sars
from evals.eval_functions import compare_sar_details
from utils import generate_dynamic_output_file_name,read_data
from agents.workflows import run_agentic_workflow1
import os
import pandas as pd
import logging
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Configure root logger for standalone script
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # File handler
    output_log_file = generate_dynamic_output_file_name('workflow1_eval', output_file_type="log", output_folder="./logs")
    file_handler = logging.FileHandler(output_log_file, mode='w')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Get predictions for each SAR
    config_file = 'configs/agents_config.yaml' 
    
    predicted_sar_details = []
    for idx, sar in enumerate(sars):
        logger.info(f"Getting Predictions for SAR {sar.sar_name}...")
        # Run the agent workflow
        pred_output = run_agentic_workflow1(sar.sar_narrative, config_file)
        predicted_sar_details.append(pred_output)

    logger.info("Evaluating Predicted SAR Details")
    entity_metrics, narrative_metrics = compare_sar_details(sars,predicted_sar_details)
    # Add execution timestamp
    entity_metrics['timestamp'] = pd.Timestamp.now()
    narrative_metrics['timestamp'] = pd.Timestamp.now()

    output_folder = "./data/output/evals/workflow1"
     # Print the DataFrames in the console
    logger.info(f"\n=== Per-SAR Entity/Account Metrics available in {output_folder}===")
    print(entity_metrics.to_string(index=False))

    output_file = generate_dynamic_output_file_name(filename="entity_metrics",output_file_type="csv",output_folder=output_folder)
    entity_metrics.to_csv(output_file)

    logger.info(f"\n=== Narrative Match Results available in {output_folder} ===")
    #print(narrative_metrics.to_string(index=False))
    output_file = generate_dynamic_output_file_name(filename="narrative_metrics",output_file_type="csv",output_folder=output_folder)
    narrative_metrics.to_csv(output_file)

    # Accumulate to master DataFrames for trend analysis
    master_entity_file = os.path.join(output_folder, "master_entity_metrics.csv")
    if os.path.exists(master_entity_file):
        master_entity_df = pd.read_csv(master_entity_file, parse_dates=["timestamp"])
        master_entity_df = pd.concat([master_entity_df, entity_metrics], ignore_index=True)
    else:
        master_entity_df = entity_metrics.copy()
    master_entity_df.to_csv(master_entity_file, index=False)

    master_narrative_file = os.path.join(output_folder, "master_narrative_metrics.csv")
    if os.path.exists(master_narrative_file):
        master_narrative_df = pd.read_csv(master_narrative_file, parse_dates=["timestamp"])
        master_narrative_df = pd.concat([master_narrative_df, narrative_metrics], ignore_index=True)
    else:
        master_narrative_df = narrative_metrics.copy()
    master_narrative_df.to_csv(master_narrative_file, index=False)
    