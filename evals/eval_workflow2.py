import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
from agents.workflows import run_agentic_workflow2
from utils import read_data , generate_dynamic_output_file_name
from evals.eval_functions import compare_trxns
from golden_data import sars, expected_trxns
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config_file = 'configs/agents_config.yaml' 
sar_narratives = read_data(train = True)


if __name__ == "__main__":
    #Restrict to just first sar for testing
    sars = sars[0]

    # Get predicted trxns for each sar extract   
    sar_trxn_metrics = []
    for idx, sar in enumerate(sars):
        sar_name = sar[sar_name]
        logging(f"Getting Predictions for SAR {idx+1}/{len(sars)}...")


        # Run the agent workflow
        pred_output = run_agentic_workflow2(sar.get_sar_extract(), config_file)
        #Aggregate list of dictionaries into a dataframe

        logging(f"Evaluating Predictions for SAR {idx+1}/{len(sars)}...")
        trxn_metrics = compare_trxns(pred_output,expected_trxns[sar_name])
        sar_trxn_metrics.append(pred_output)

    sar_trxn_metrics_df = pd.concat(sar_trxn_metrics,ignore_index= True)


    output_folder = "./data/output/evals/workflow2"
     # Print the DataFrames in the console
    logging(f"\n=== Per-SAR Entity/Account Metrics available in {output_folder}===")
    #print(entity_metrics.to_string(index=False))

    output_file = generate_dynamic_output_file_name(filename="trxn_metrics",output_file_type="csv",output_folder=output_folder)
    sar_trxn_metrics_df.to_csv(output_file)
    