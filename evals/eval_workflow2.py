import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
from agents.workflows import run_agentic_workflow2
from utils import read_data , generate_dynamic_output_file_name
from evals.eval_functions import compare_trxns
from evals.golden_data import sars, expected_trxns
import logging
# Configure logging to both file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
output_log_file = generate_dynamic_output_file_name('workflow2_eval', output_file_type="log", output_folder="./logs")
file_handler = logging.FileHandler(output_log_file, mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

config_file = 'configs/agents_config.yaml' 
sar_narratives = read_data(train = True)  


if __name__ == "__main__":
    #Restrict to just first sar for testing
    sars = sars[:1]
    
    # Get predicted trxns for each sar extract   
    sar_trxn_metrics = []
    for idx, sar in enumerate(sars):
        sar_name = sar.sar_name
        logging.info(f"Getting Predictions for SAR {idx+1}/{len(sars)}...")
        # print(f"sar_{idx}: \n {sar.get_sar_extract()}")

        # Run the agent workflow
        pred_output = run_agentic_workflow2(sar.get_sar_extract(), config_file)
        #pred_output = pd.read_csv("./data/output/results_trxn_metrics_20250412_021639.csv")

        logging.info(f"Evaluating Predictions for SAR {idx+1}/{len(sars)}...")
       
        trxn_metrics = compare_trxns(pred_output,expected_trxns)
        sar_trxn_metrics.append(trxn_metrics)

    sar_trxn_metrics_df = pd.concat(sar_trxn_metrics,ignore_index= True)

    output_folder = "./data/output/evals/workflow2"
     # Print the DataFrames in the console
    logging.info(f"\n=== Per-SAR Trxn Metrics available in {output_folder}===")
    #print(entity_metrics.to_string(index=False))

    output_file = generate_dynamic_output_file_name(filename="trxn_metrics",output_file_type="csv",output_folder=output_folder)
    sar_trxn_metrics_df.to_csv(output_file)