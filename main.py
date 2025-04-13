# main.py

from utils import read_file, write_data_to_file, generate_dynamic_output_file_name
from agents.workflows import run_agentic_workflow1, run_agentic_workflow2
from dotenv import load_dotenv
import os
import logging



logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')




def main(filename):
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")
    config_file = 'configs/agents_config.yaml'  # Update this path as needed
    
    #train_sars = read_data(train=True)
    #Read specific SAR
    message = read_file(filename)

    # Run first agentic workflow to extract entities
    results1 = run_agentic_workflow1(message,config_file)

    #Loop through the narrative for each account and generate transactions - To be Done
    ###For now just pick one account for testing purposes.
    #results1_p1 = {key: (value if key != 'Narratives' else {'345723': value['345723']}) for key, value in results1.items()}



    
    # results1_p1 = {'Entities': 
    #                         {'Individuals': ['John', 'Jill'], 
    #                         'Organizations': ['Acme Inc'], 
    #                         'Financial_Institutions': ['Bank of America', 'Chase Bank']},
    #                   'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
    #                   'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
    #                   'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
    #                   'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
    #                   'Narratives' : {"345723": 
    #                          {
    #                           "Trxn_Set_1":"John sent 25 wires to Acct #98765 between Jan 10,2025 and Feb 15, 2025. The trxns ranged from $1,000 to $5,000"} }
    #                  }

    # Run second agentic worklfow to extract transactions
    results2 = run_agentic_workflow2(results1,config_file)
    
    

    return results2 
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read and process a file.")
    parser.add_argument("filename", help="Name of the file to be read and processed")
    args = parser.parse_args()
    main(args.filename)
