import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
from agents.workflows import run_agentic_workflow1
from utils import read_data
from evals.eval_functions import compare_trxns



# ============================
# Define expected transactions for each SAR
# ============================


# Define a data structure for each SAR
expected_trxns = {
"SAR_1": 
    {"Trxn_Set_1":
         {"Originator_Account_ID": "12345-6789",
          "Beneficiary_Account_ID": "12345-6789",
          "Total_Amount": 50000,
          "Trxn_Type": ["Cash","Check","Money Order"],
          "Min_Date": "2003-01-02",
          "Max_Date": "2003-03-17",
          "Branch_ATM_Location": [],
          "Min_Ind_Amt":1500,
          "Max_Ind_Amt":9500,
          "N_trxns": 13}
        ,
     "Trxn_Set_2":
        {"Originator_Account_ID": "12345-6789",
          "Beneficiary_Account_ID": "3489728",
          "Total_Amount": 225000,
          "Trxn_Type": ["Wire"],
          "Min_Date": "2003-01-17",
          "Max_Date": "2003-03-21",
          "Branch_ATM_Location": [],
          "Min_Ind_Amt":25000,
          "Max_Ind_Amt":25000,
          "N_trxns" : 9
        },

     "Trxn_Set_3":
        {"Originator_Account_ID": "23456-7891",
          "Beneficiary_Account_ID": "23456-7891",
          "Total_Amount": 275000,
          "Trxn_Type": ["Cash","Check","Money Order"],
          "Min_Date": "2003-01-02",
          "Max_Date": "2003-03-17",
          "Min_Ind_Amt":4400,
          "Max_Ind_Amt":9900,
          "N_trxns" : 33
        }
    
    },

# "SAR_2": {
#     "Trxn_Set_1":
#          {"Originator_Account_ID": "1234567",
#           "Beneficiary_Account_ID": "1234567",
#           "Total_Amount": 58800,
#           "Trxn_Type": ["Cash"],
#           "Min_Date": "2003-06-03",
#           "Max_Date": "2003-03-12",
#           "Branch_ATM_Location": ["Happy Valley branch","Main office branch"],
#           "Min_Ind_Amt": 9800,
#           "Max_Ind_Amt":9800,
#           "N_trxns": 6},

#     "Trxn_Set_2":
#          {"Originator_Account_ID": "1234567",
#           "Beneficiary_Account_ID": "Dummy_Acct_1",
#           "Total_Amount": 58800,
#           "Trxn_Type": ["Check"],
#           "Min_Date": "2003-06-04",
#           "Max_Date": "2003-06-13",
#           "Branch_ATM_Location": [],
#           "Min_Ind_Amt": 9800,
#           "Max_Ind_Amt":9800,
#           "N_trxns": 6}

# },

# "SAR_3": {
#      "Trxn_Set_1":
#          {"Originator_Account_ID": "12345678910",
#           "Beneficiary_Account_ID": "12345678910",
#           "Total_Amount": 29650,
#           "Trxn_Type": ["Cash"],
#           "Min_Date": "2002-03-15",
#           "Max_Date": "2003-03-18",
#           "Branch_ATM_Location": [],
#           "Min_Ind_Amt": 9700,
#           "Max_Ind_Amt":9900,
#           "N_trxns": 3},

#     "Trxn_Set_2":
#          {"Originator_Account_ID": "12345678910",
#           "Beneficiary_Account_ID": "981012345",
#           "Total_Amount": 29500,
#           "Trxn_Type": ["Wire"],
#           "Min_Date": "2002-03-16",
#           "Max_Date": "2002-03-19",
#           "Branch_ATM_Location": [],
#           "Min_Ind_Amt": 9700,
#           "Max_Ind_Amt": 9900,
#           "N_trxns": 3}
#     },


# "SAR_4": {
#      "Trxn_Set_1":
#          {"Originator_Account_ID": "54321098",
#           "Beneficiary_Account_ID": "54321098",
#           "Total_Amount": 2710000,
#           "Trxn_Type": ["Cash"],
#           "Min_Date": "1999-02-02",
#           "Max_Date": "2001-06-20",
#           "Branch_ATM_Location": ["North Burlington", "South Burlington", "West Burlington"],
#           "Min_Ind_Amt": 0.01,
#           "Max_Ind_Amt": 2710000,
#           "N_trxns": 284},

#     "Trxn_Set_2":
#          {"Originator_Account_ID": "54321098",
#           "Beneficiary_Account_ID": "456781234",
#           "Total_Amount": 2697000,
#           "Trxn_Type": ["Wire"],
#           "Min_Date": "1999-02-03",
#           "Max_Date": "2001-06-21",
#           "Branch_ATM_Location": [],
#           "Min_Ind_Amt": 9700,
#           "Max_Ind_Amt": 9900,
#           "N_trxns": 274}
#     },
#     "Trxn_Set_3":
#          {"Originator_Account_ID": "12345678",
#           "Beneficiary_Account_ID": "12345678",
#           "Total_Amount": 1900000,
#           "Trxn_Type": ["Cash"],
#           "Min_Date": "1999-02-02",
#           "Max_Date": "2001-06-20",
#           "Branch_ATM_Location": [],
#           "Min_Ind_Amt": 8720,
#           "Max_Ind_Amt": 16500,
#           "N_trxns": 200},

#     "Trxn_Set_4":
#          {"Originator_Account_ID": "12345678",
#           "Beneficiary_Account_ID": "456781234",
#           "Total_Amount": 1866000,
#           "Trxn_Type": ["Wire"],
#           "Min_Date": "1999-02-03",
#           "Max_Date": "2001-06-21",
#           "Branch_ATM_Location": [],
#           "Min_Ind_Amt": 8720,
#           "Max_Ind_Amt": 16500,
#           "N_trxns": 200},

#     "Trxn_Set_5":
#          {"Originator_Account_ID": "689472",
#           "Beneficiary_Account_ID": "12345678",
#           "Total_Amount": 100000,
#           "Trxn_Type": ["Wire"],
#           "Min_Date": "1999-02-03",
#           "Max_Date": "2001-06-21",
#           "Branch_ATM_Location": [],
#           "Min_Ind_Amt": 1000,
#           "Max_Ind_Amt": 100000,
#           "N_trxns": 200}
    }


config_file = 'configs/agents_config.yaml' 
sar_narratives = read_data(train = True)


if __name__ == "__main__":
    

    trxn_metrics = compare_trxns(sars)

     # Print the DataFrames in the console
    print("\n=== Per-SAR Entity/Account Metrics ===")
    print(entity_metrics.to_string(index=False))

    print("\n=== Narrative Match Results ===")
    print(narrative_metrics.to_string(index=False))
    