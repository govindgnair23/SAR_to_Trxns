import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
from agents.workflows import run_agentic_workflow1
from utils import read_data
from evals.eval_functions import evaluate_sars
# ============================
# Define SARs and Gold Standards
# ============================

# Example SARs and their gold standard outputs
# In practice, you would load this data from a file or database

# Define a data structure for each SAR
class SAR:
    def __init__(self, sar_name: str, sar_narrative: str, gold_entities: Dict[str, List[str]],
                 gold_account_ids: List[str],
                 gold_acct_to_fi: Dict[str, str],
                 gold_acct_to_cust: Dict[str, str],
                 gold_fi_to_acct_to_cust: Dict[str, Dict[str, str]],
                 gold_narrative: Dict[str,str]
                 ):
        self.sar_name = sar_name
        self.sar_narrative = sar_narrative,
        self.gold_entities = gold_entities
        self.gold_account_ids = gold_account_ids
        self.gold_acct_to_fi = gold_acct_to_fi
        self.gold_acct_to_cust = gold_acct_to_cust
        self.gold_fi_to_acct_to_cust = gold_fi_to_acct_to_cust
        self.gold_narrative = gold_narrative

config_file = 'configs/agents_config.yaml' 
sar_narratives = read_data(train = True)

# Example SARs (Add more SAR instances as needed)
sars = [
    SAR(
        sar_name="sar_train1",
        sar_narrative = sar_narratives[0],
        gold_entities={
            "Individuals": ["John Doe"],
            "Organizations": ["Acme, Inc.", "Kulkutta Building Supply Company"],
            "Financial Institutions": ["Bank of Anan"]
        },
        gold_account_ids=["123456789", "234567891", "3489728"],
        gold_acct_to_fi={
            "123456789": "Dummy_Bank_1",
            "234567891": "Dummy_Bank_1",
            "3489728": "Bank of Anan"
        },
        gold_acct_to_cust={
            "123456789": "John Doe",
            "234567891": "Acme, Inc.",
            "3489728": "Kulkutta Building Supply Company"
        },
        gold_fi_to_acct_to_cust = {'Dummy_Bank_1': {'12345-6789': 'CUST_001',
                                                   '23456-7891': 'CUST_002'},
                              'Bank of Anan': {'3489728': 'CUST_003'}},

        gold_narrative =  {
            '12345-6789': 'John Doe opened a personal checking account, #12345-6789, in March of 1994. Between January 2 and March 17, 2003, 13 deposits totaling approximately $50,000 were posted to the account, consisting of cash, checks, and money orders, with amounts ranging from $1,500 to $9,500. Third-party out of state checks and money orders were also deposited. Between January 17, 2003, and March 21, 2003, John Doe originated nine wires totaling $225,000 to the Bank of Anan in Dubai, UAE, to benefit Kulkutta Building Supply Company, account #3489728.',
                        
            '23456-7891': 'A business checking account, #23456-7891, for Acme, Inc. was opened in January of 1998. Between January 2 and March 17, 2003, 33 deposits totaling approximately $275,000 were made to the account, consisting of cash, checks, and money orders. Individual amounts ranged between $4,446 and $9,729; 22 of 33 deposits were between $9,150 and $9,980. In nine instances where cash deposits were made to both accounts on the same day, combined deposits exceeded $10,000. Currency transaction reports were filed with the IRS for daily transactions exceeding $10,000. The bank identified Acme, Inc. as providing remittance services to the Middle East, including Iran, without being a licensed money wire transfer business.',
                        
            '3489728': "Nine wire transfers totaling $225,000 were sent from John Doe's personal account #12345-6789 at Dummy_Bank_1 to Kulkutta Building Supply Company, account #3489728 at the Bank of Anan in Dubai, UAE, between January 17, 2003, and March 21, 2003."}
    ),
    # Add second SAR
    # SAR(
    #     sar_name="sar_train2",
    #     sar_narrative = sar_narratives[1],
    #     gold_entities={
    #        "Individuals": ["John Doe", "Jane Doe"],
    #         "Organizations": ["Doe’s Auto Sales"],
    #         "Financial Institutions": ["XYZ Bank"]
    #     },
    #     gold_account_ids= ["1234567", "Dummy_Acct_1"],
    #     gold_acct_to_fi={
    #                 "1234567": "Dummy_Bank_1" ,
    #                 "Dummy_Acct_1": "XYZ Bank"
    #             },
    #     gold_acct_to_cust= {
    #             "1234567": "Doe’s Auto Sales",
    #             "Dummy_Acct_1": "Doe’s Auto Sales"
    #                    }
    # ),
    # # Add third SAR
    # SAR(
    #     sar_name="sar_train3",
    #     sar_narrative = sar_narratives[2],
    #     gold_entities={
    #                 "Individuals": ["John Doe", "Jennifer Doe"],
    #                 "Organizations": ["Quickie Car Wash"],
    #                 "Financial Institutions": ["Aussie Bank"]
    #             },
    #     gold_account_ids=  ["12345678910", "981012345"],
    #     gold_acct_to_fi={
    #                 "12345678910": "Dummy_Bank_1" ,
    #                 "981012345": "Aussie Bank"
    #             },
    #         gold_acct_to_cust=  {
    #                     "12345678910": "John Doe",
    #                     "981012345": "Jennifer Doe"
    #                 }
    # ),
    # # Add fourth SAR
    # SAR(
    #     sar_name="sar_train4",
    #     sar_narrative = sar_narratives[3],
    #     gold_entities={
    #                     "Individuals": ["Paul Lafonte"],
    #                     "Organizations": ["Sky Corporation", "Sea Corporation", "Tolinka Inc."],
    #                     "Financial Institutions": ["Bank of Mainland", "Bank XYZ", "Bank of Poland", "Artsy Bank"]
    #                 },
    #     gold_account_ids=   ["54321098", "12345678", "689472", "456781234", "Dummy_Acct_1","Dummy_Acct_2"],
    #     gold_acct_to_fi={
    #                 "54321098": "Bank of Mainland",
    #                 "12345678": "Bank of Mainland",
    #                 "689472": "Bank XYZ",
    #                 "456781234": "Artsy Bank",
    #                 "Dummy_Acct_1": "Bank of Poland",
    #                  "Dummy_Acct_2": "Bank of Mainland"
    #             },
    #     gold_acct_to_cust=  {
    #                     "54321098": "Sky Corporation",
    #                     "12345678": "Sea Corporation",
    #                     "689472": "Tolinka Inc.",
    #                     "456781234": "Paul Lafonte",
    #                     "Dummy_Acct_1": "Bank XYZ",
    #                     "Dummy_Acct_2": "Bank of Poland"
    #                 }
    # ),

]

if __name__ == "__main__":
    entity_metrics, narrative_metrics = evaluate_sars(sars)

     # Print the DataFrames in the console
    print("\n=== Per-SAR Entity/Account Metrics ===")
    print(entity_metrics.to_string(index=False))

    print("\n=== Narrative Match Results ===")
    print(narrative_metrics.to_string(index=False))
    