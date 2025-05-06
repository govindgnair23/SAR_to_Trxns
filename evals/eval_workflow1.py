
from evals.golden_data import sars
from evals.eval_functions import compare_sar_details
from utils import generate_dynamic_output_file_name,read_data
from agents.workflows import run_agentic_workflow1
import logging
# Configure logging
# Configure logging to both file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
output_log_file = generate_dynamic_output_file_name('workflow1_eval', output_file_type="log", output_folder="./logs")
file_handler = logging.FileHandler(output_log_file, mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if __name__ == "__main__":
    # Get predictions for each SAR

    config_file = 'configs/agents_config.yaml' 
    
    predicted_sar_details = []
    for idx, sar in enumerate(sars):
        logging.info(f"Getting Predictions for SAR {sar.sar_name}...")
        # Run the agent workflow
        pred_output = run_agentic_workflow1(sar.sar_narrative, config_file)
        predicted_sar_details.append(pred_output)

    logging.info("Evaluating Predicted SAR Details")
    entity_metrics, narrative_metrics = compare_sar_details(sars,predicted_sar_details)

    output_folder = "./data/output/evals/workflow1"
     # Print the DataFrames in the console
    logging.info(f"\n=== Per-SAR Entity/Account Metrics available in {output_folder}===")
    print(entity_metrics.to_string(index=False))

    output_file = generate_dynamic_output_file_name(filename="entity_metrics",output_file_type="csv",output_folder=output_folder)
    entity_metrics.to_csv(output_file)

    logging.info(f"\n=== Narrative Match Results available in {output_folder} ===")
    #print(narrative_metrics.to_string(index=False))
    output_file = generate_dynamic_output_file_name(filename="narrative_metrics",output_file_type="csv",output_folder=output_folder)
    narrative_metrics.to_csv(output_file)
    