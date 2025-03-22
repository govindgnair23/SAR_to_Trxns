
from golden_data import sars
from evals.eval_functions import compare_sar_details
from utils import generate_dynamic_output_file_name,read_data
from agents.workflows import run_agentic_workflow1
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Get predictions for each SAR

    config_file = 'configs/agents_config.yaml' 
    
    predicted_sar_details = []
    for idx, sar in enumerate(sars):
        logging(f"Getting Predictions for SAR {idx+1}/{len(sars)}...")
        # Run the agent workflow
        pred_output = run_agentic_workflow1(sar.sar_narrative, config_file)
        predicted_sar_details.append(pred_output)

    logging("Evaluating Predicted SAR Details")
    entity_metrics, narrative_metrics = compare_sar_details(sars,predicted_sar_details)

    output_folder = "./data/output/evals/workflow1"
     # Print the DataFrames in the console
    logging(f"\n=== Per-SAR Entity/Account Metrics available in {output_folder}===")
    #print(entity_metrics.to_string(index=False))

    output_file = generate_dynamic_output_file_name(filename="entity_metrics",output_file_type="csv",output_folder=output_folder)
    entity_metrics.to_csv(output_file)

    logging(f"\n=== Narrative Match Results available in {output_folder} ===")
    #print(narrative_metrics.to_string(index=False))
    output_file = generate_dynamic_output_file_name(filename="narrative_metrics",output_file_type="csv",output_folder=output_folder)
    narrative_metrics.to_csv(output_file)
    