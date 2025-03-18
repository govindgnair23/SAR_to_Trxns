import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
from agents.workflows import run_agentic_workflow1
from utils import read_data, generate_dynamic_output_file_name
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
            '12345-6789': 
                {"Trxn_Set_1": "John Doe opened a personal checking account, #12345-6789, in March of 1994. Between January 2 and March 17, 2003, 13 deposits totaling approximately $50,000 were posted to the account, consisting of cash, checks, and money orders, with amounts ranging from $1,500 to $9,500. Third-party out of state checks and money orders were also deposited.",
                
                "Trxn_Set_2": "Between January 17, 2003, and March 21, 2003, John Doe originated nine wires totaling $225,000 to the Bank of Anan in Dubai, UAE, to benefit Kulkutta Building Supply Company, account #3489728. The wire transfers were always  conducted at the end of each week in the amount of $25,000."},
                        
            '23456-7891':
                {"Trxn_Set_1":"A business checking account, #23456-7891, for Acme, Inc. was opened in January of 1998. Between January 2 and March 17, 2003, 33 deposits totaling approximately $275,000 were made to the account, consisting of cash, checks, and money orders. Individual amounts ranged between $4,446 and $9,729; 22 of 33 deposits were between $9,150 and $9,980. In nine instances where cash deposits were made to both accounts on the same day, combined deposits exceeded $10,000. Currency transaction reports were filed with the IRS for daily transactions exceeding $10,000. The bank identified Acme, Inc. as providing remittance services to the Middle East, including Iran, without being a licensed money wire transfer business."},
                        
            '3489728': 
                {"Trxn_Set_1":"Nine wire transfers totaling $225,000 were sent from John Doe's personal account #12345-6789 at Dummy_Bank_1 to Kulkutta Building Supply Company, account #3489728 at the Bank of Anan in Dubai, UAE, between January 17, 2003, and March 21, 2003."
                       }}
    ),

    ## Add Second SAR
    SAR(
        sar_name="sar_train2",
        sar_narrative = sar_narratives[1],
        gold_entities={
              "Individuals": ["John Doe","Jane Doe"],
              "Organizations": ["Doe's Auto Sales"],
              "Financial_Institutions": ["XYZ Bank"]
        },
        gold_account_ids=["1234567"],
        gold_acct_to_fi={
            "1234567": "Dummy_Bank_1",
            "Dummy_Acct_1": "XYZ Bank"
        },
        gold_acct_to_cust={
           "1234567": "Doe's Auto Sales",
           "Dummy_Acct_1": "XYZ Bank"
        },
        gold_fi_to_acct_to_cust = { "Dummy_Bank_1": {"1234567": "CUST_001"},
        "XYZ Bank": {"Dummy_Acct_1":"CUST_002"}
                 },

        gold_narrative =  {
             "1234567": {
                    "Trxn_Set_1":"The account #1234567 for Doe's Auto Sales shows unusual activity characterized by structured cash deposits. On six occasions in June 2003, cash deposits of $9,800 were made, possibly to avoid the filing of a currency transaction report. Deposits were made by John Doe on 06/03, 06/09, and 06/11 at the Happy Valley branch, while Jane Doe made deposits on 06/04, 06/10, and 06/12 at the Main Office branch",

                    "Trxn_Set_2": "Following these deposits, checks for $9,800 were issued and subsequently deposited at XYZ Bank on 06/04, 06/05, 06/10, 06/11, 06/12, and 06/13. The source of the cash is unknown, and this pattern appears to evade the reporting requirements of the Bank Secrecy Act."}
                        }
    ),

    ## Add third SAR
    SAR(
        sar_name="sar_train3",
        sar_narrative = sar_narratives[2],
        gold_entities={
              "Individuals": ["John Doe","Jane Doe"],
              "Organizations": ["Quickie Car Wash"],
              "Financial_Institutions": ["Aussie Bank"]
        },
        gold_account_ids=["12345678910","981012345"],
        gold_acct_to_fi={
            "12345678910": "Dummy_Bank_1",
            "981012345": "Aussie Bank"
        },
        gold_acct_to_cust={
           "12345678910": "John Doe",
           "981012345": "Jennifer Doe"
        },
        gold_fi_to_acct_to_cust = {
        "Dummy_Bank_1": {
            "12345678910": "CUST_001"
        },
        "Aussie Bank": {
            "981012345": "CUST_002"
        }
            },

        gold_narrative =  {
                 "12345678910": 
                    {"Trxn_Set_1": "John Doe made structured cash deposits into his personal checking account at Dummy_Bank_1, totaling $29,650, with the following details: 03/15/02 - $9,950.00; 03/17/02 - $9,700.00; 03/18/02 - $10,000.",
                     "Trxn_Set_2": "Following the structured cash deposits, John Doe  conducted immediate wire transfers to Jennifer Doe's account at Aussie Bank, totaling $29,500. The wire transfer details are as follows: 03/16/02 - $9,900.00; 03/18/02 - $9,700.00; 03/19/02 - $9,900.00."},
                 "981012345": 
                      {"Trxn_Set_1": "Jennifer Doe receives wire transfers from John Doe's account at Dummy_Bank_1, with total amounts received being $29,500 over several transactions: 03/16/02 - $9,900.00; 03/18/02 - $9,700.00; 03/19/02 - $9,900.00."}
                        }
    ),

    # Add Fourth SAR

    SAR(
        sar_name="sar_train4",
        sar_narrative = sar_narratives[3],
        gold_entities={
                "Individuals": ["Paul Lafonte"],
        "Organizations": [
            "Sky Corporation",
            "Sea Corporation",
            "Tolinka Inc."
        ],
        "Financial_Institutions": [
            "Bank of Mainland",
            "Bank XYZ",
            "Bank of Poland",
            "Artsy Bank"
        ]
        },
        gold_account_ids=[  "54321098","12345678","689472","Dummy_Acct_1"],
        gold_acct_to_fi={
            "54321098": "Bank of Mainland",
            "12345678": "Bank of Mainland",
            "689472": "Bank XYZ",
            "Dummy_Acct_1": "Dummy_Bank_1"
        },
        gold_acct_to_cust={
           "54321098": "Sky Corporation",
            "12345678": "Sea Corporation",
           "689472": "Tolinka Inc.",
           "Dummy_Acct_1": "Dummy_Customer_1"
        },
        gold_fi_to_acct_to_cust = {
        "Bank of Mainland": {
            "54321098": "Sky Corporation",
            "12345678": "Sea Corporation"
        },
        "Bank XYZ": {
            "689472": "Tolinka Inc."
        },
        "Dummy_Bank_1": {
            "Dummy_Acct_1": "Dummy_Customer_1"
        }
            },

        gold_narrative =  {
                 
        "54321098": 
           { "Trxn_Set_1": "Between  2/2/99 through 6/20/01 ,Sky Corporation had 284 cash deposits totaling $2,710,000 at Bank of Mainland, conducted through three main branches: North Burlington, South Burlington, and West Burlington. The average deposit amount ranged from $8,720 to $16,500. ",
            "Trxn_Set_2": "Between  2/2/99 through 6/20/01, the day after deposits, 274 outgoing wire transfers totaling $2,697,000 were conducted, typically sent to Paul Lafonte at Artsy Bank, account #456781234 in Paris, France, using a remote computer terminal.The activity occured for the period  2/2/99 through 6/20/01"},

        "12345678": {
            "Trxn_Set_1": " Between  2/2/99 through 6/20/01, Sea Corporation had 200 cash deposits totaling $1,900,000 at Bank of Mainland, with many transactions conducted on the same day at multiple branches to potentially circumvent federal reporting requirements. ",

            "Trxn_Set_2": "The company processed 198 outgoing wire transfers totaling $1,866,000, usually mirroring the deposits from the previous day, and also directed to Paul Lafonte at Artsy Bank, account #456781234 in Paris, France.The activity occured for the period  2/2/99 through 6/20/01"},

        "689472": {
            "Trxn_Set_1": "Tolinka Inc. is a customer of Bank XYZ and received 15 incoming wire transfers affecting account #12345678 at Bank of Mainland. Bank XYZ was unable to provide documentation for Tolinka Inc., and after inquiries, Tolinka Inc. closed its account without explanation.The activity occured for the period  2/2/99 through 6/20/01"}
    }

    )


    
]

if __name__ == "__main__":
    entity_metrics, narrative_metrics = evaluate_sars(sars)

    output_folder = "./data/output/evals/workflow1"
     # Print the DataFrames in the console
    print(f"\n=== Per-SAR Entity/Account Metrics available in {output_folder}===")
    #print(entity_metrics.to_string(index=False))

    output_file = generate_dynamic_output_file_name(filename="entity_metrics",output_file_type="csv",output_folder=output_folder)
    entity_metrics.to_csv(output_file)

    print(f"\n=== Narrative Match Results available in {output_folder} ===")
    #print(narrative_metrics.to_string(index=False))
    output_file = generate_dynamic_output_file_name(filename="narrative_metrics",output_file_type="csv",output_folder=output_folder)
    narrative_metrics.to_csv(output_file)
    