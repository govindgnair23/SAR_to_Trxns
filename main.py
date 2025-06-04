# main.py

from utils import read_file, write_data_to_file, generate_dynamic_output_file_name
from agents.workflows import run_agentic_workflow1, run_agentic_workflow2
from dotenv import load_dotenv
import os
import logging
# Configure logging to both file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
output_log_file = generate_dynamic_output_file_name('main', output_file_type="log", output_folder="./logs")
file_handler = logging.FileHandler(output_log_file, mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)




def main(filename):
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")
    config_file = 'configs/agents_config.yaml'  # Update this path as needed
    
    #train_sars = read_data(train=True)
    #Read specific SAR
    
    #message = read_file(filename)
    #logging.info("Read SAR")

    # Run first agentic workflow to extract entities
    #results1 = run_agentic_workflow1(message,config_file)
    #logging.info("Ran first workflow to extract entities and narratives")

    #Loop through the narrative for each account and generate transactions - To be Done
    ###For now just pick one account for testing purposes.
    #results1_p1 = {key: (value if key != 'Narratives' else {'345723': value['345723']}) for key, value in results1.items()}



    
    results1_p1 = {'Entities': 
                            {'Individuals': ['John', 'Jill'], 
                            'Organizations': ['Acme Inc'], 
                            'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                      'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                      'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                      'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                      'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                      'Narratives' : {"345723": 
                             {
                              "Trxn_Set_1":"John sent 2 wires to Acct #98765 on Jan 10,2025 and Feb 15, 2025. The trxns were for $1,000 and $5,000"} }
                     }

    # Run second agentic worklfow to extract transactions
    results2 = run_agentic_workflow2(results1_p1,config_file)
    logging.info("Ran second workflow to generate transactions")
    
    

    return results2 
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read and process a file.")
    parser.add_argument("filename", help="Name of the file to be read and processed")
    args = parser.parse_args()
    main(args.filename)
