import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
from agents.workflows import run_agentic_workflow2
from utils import read_data , generate_dynamic_output_file_name,write_data_to_file
from evals.eval_functions import compare_trxns
from evals.golden_data import sars, expected_trxns
import logging
import time
logger = logging.getLogger(__name__)

config_file = 'configs/agents_config.yaml' 



if __name__ == "__main__":
    # Configure root logger for standalone script
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # File handler
    output_log_file = generate_dynamic_output_file_name('workflow2_eval', output_file_type="log", output_folder="./logs")
    file_handler = logging.FileHandler(output_log_file, mode='w')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
  
    #Restrict to just first sar for testing
    #sars = [sars[1]]
    
    # Get predicted trxns for each sar extract   
    sar_trxn_metrics = []
    sar_timings = []
    total_start_time = time.time()
    for idx, sar in enumerate(sars):
        # Start timing this SAR iteration
        start_time = time.time()
        
        logger.info(f"Getting Predictions for SAR {sar.sar_name}...")
        # print(f"sar_{idx}: \n {sar.get_sar_extract()}")

        # Run the agent workflow
        pred_output = run_agentic_workflow2(sar.get_sar_extract(), config_file)
        #pred_output = pd.read_csv("./data/output/results_trxn_metrics_20250412_021639.csv")

        logger.info(f"Evaluating Predictions for SAR {sar.sar_name}...")
       
        trxn_metrics = compare_trxns(pred_output,expected_trxns[sar.sar_name])
        trxn_metrics.insert(0, "sar_id", sar.sar_name)
        # Record elapsed time for this iterationgit a
        elapsed_time = time.time() - start_time
        sar_timings.append({"sar_id": sar.sar_name, "time_taken_seconds": elapsed_time})
        # Record elapsed time for this iteration
        elapsed_time = time.time() - start_time
        sar_timings.append({"sar_id": sar.sar_name, "time_taken_seconds": elapsed_time})
        sar_trxn_metrics.append(trxn_metrics)

    total_elapsed_time = time.time() - total_start_time
    sar_timings.append({"sar_id": "Total","time_taken_seconds": total_elapsed_time})
    sar_trxn_metrics_df = pd.concat(sar_trxn_metrics,ignore_index= True)
    sar_timings_df = pd.DataFrame(sar_timings)

    output_folder = "./data/output/evals/workflow2"
     # Print the DataFrames in the console
    logger.info(f"\n=== Per-SAR Trxn Metrics available in {output_folder}===")
    #print(entity_metrics.to_string(index=False))

    output_file = generate_dynamic_output_file_name(filename="trxn_metrics",output_file_type="csv",output_folder=output_folder)
    write_data_to_file(sar_trxn_metrics_df,output_file)

    timing_file = generate_dynamic_output_file_name(filename="sar_timings", output_file_type="csv", output_folder=output_folder)
    write_data_to_file(sar_timings_df, timing_file)