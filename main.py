# main.py

from utils import read_file, write_data_to_file, generate_dynamic_output_file_name
from agents.workflows import run_agentic_workflow1, run_agentic_workflow2
from dotenv import load_dotenv
import os
import logging
import pandas as pd
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


def generate_transactions_from_text(sar_text: str) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY")
    config_file = 'configs/agents_config.yaml'

    #Simulate loading/parsing SAR for demo
    #Replace this with actual parser for SAR -> entities
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
    #                           "Trxn_Set_1":"John sent 2 wires to Acct #98765 on Jan 10,2025 and Feb 15, 2025. The trxns were for $1,000 and $5,000"} }
    #                  }
    results1 = run_agentic_workflow1(sar_text=sar_text,config_file=config_file)
    logger.info("Ran first workflow to extract entites")
    results2 = run_agentic_workflow2(input =results1, config_file=config_file)
    logger.info("Ran second workflow to generate transactions")
    return results1,results2


def main(filename):
    sar_text = read_file(filename)
    logger.info("Read SAR file")
    _,df = generate_transactions_from_text(sar_text)
    #write_data_to_file(df,)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read and process a file.")
    parser.add_argument("filename", help="Name of the file to be read and processed")
    args = parser.parse_args()
    main(args.filename)
