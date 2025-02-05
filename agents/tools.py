from typing import List, Literal
from typing_extensions import Annotated

import numpy as np
from datetime import datetime, timedelta
import random

Channels_allowed = Literal["Wire","Cash","Check"]
def generate_transactions(
        Originator_Name:Annotated[str, "Entity or Customer originating the transactions"],
        Originator_Account_ID:Annotated[str, "Account  of Entity or Customer originating the transactions"],
        Originator_Customer_ID:Annotated[str, "Customer ID of Entity or Customer originating the transactions"],
        Beneficiary_Name:Annotated[str, "Customer ID of Entity or Customer  receiving the transactions"], 
        Beneficiary_Account_ID:Annotated[str, "Account of Entity or Customer  receiving the transactions"],
        Beneficiary_Customer_ID:Annotated[str, "Customer ID of Entity or Customer receiving the transactions"],
        Trxn_Channel:Annotated[List[Channels_allowed], "Transaction Channels used to make the transactions."],
        Start_Date:Annotated[str, "Date on which the first transaction was made"], 
        End_Date:Annotated[str, "Date on which the last transaction was made"],
        Min_Ind_Trxn_Amt:Annotated[float, "The smallest transaction amount"],
        Max_Ind_Trxn_Amt:Annotated[float, "The largest transaction amount"],
        Branch_or_ATM_Location:Annotated[str, "The location where transaction was originated or received"],
        N_transactions:Annotated[int, "The number of transactions made between the Originator and Beneficary"]) -> dict:
    '''
    Tool to generate trxns
    '''
    
    Start_Date = datetime.strptime(Start_Date,"%Y-%m-%d")
    End_Date = datetime.strptime(End_Date,"%Y-%m-%d")
    trxns = {} #Dictionary to hold transactions
    trxn_channels = random.choices(Trxn_Channel, k = N_transactions)
    
    sample_deltas  =  random.choices(range((End_Date - Start_Date).days),k = N_transactions) #Get random number of days to be added to get new dates
    trxn_dates = [   Start_Date + timedelta(delta) for delta in sample_deltas] # TO DO: Add start and end date to the list
    #Convert back to string
    trxn_dates = [trxn_date.strftime("%Y-%m-%d") for trxn_date in trxn_dates]
    trxn_amounts = np.round(np.random.uniform(low=Min_Ind_Trxn_Amt,high=Max_Ind_Trxn_Amt,size = N_transactions),2)

    for i in range(N_transactions):
        trxns[f"Trxn_{i+1}"] = {"Originator_Name": Originator_Name , "Originator_Account_ID": Originator_Account_ID,"Originator_Customer_ID": Originator_Customer_ID,
                            "Beneficiary_Name": Beneficiary_Name, "Beneficiary_Account_ID": Beneficiary_Account_ID,"Beneficiary_Customer_ID": Beneficiary_Customer_ID,
                             "Trxn_Channel": trxn_channels[i], "Trxn_Date": trxn_dates[i], "Trxn_Amount":trxn_amounts[i],
                              "Branch_or_ATM_Location": Branch_or_ATM_Location }

    return trxns