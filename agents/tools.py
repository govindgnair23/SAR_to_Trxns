from typing import List, Literal, Union
from typing_extensions import Annotated

import numpy as np
from datetime import datetime, timedelta
import random
import math
import logging

logger = logging.getLogger(__name__)

###To Do###
# Expect one of
# a) min_ind_amount, max_ind_amount and N_trxns
# b) Total Amount and N_trxns
# c) Total Amount and min_ind_amount and max_ind_amount

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
        Branch_or_ATM_Location:Annotated[Union[str, List[str]], "Branch or ATM location(s) for transactions"],
        N_transactions:Annotated[int, "The number of transactions made between the Originator and Beneficary"] = None,
        Total_Amount:Annotated[float, "Total amount of all transactions"] = None) -> dict:
    '''
    Tool to generate trxns
    '''
    
    Start_Date = datetime.strptime(Start_Date,"%Y-%m-%d")
    End_Date = datetime.strptime(End_Date,"%Y-%m-%d")
    
    # Determine scenario and derive N_transactions if needed
    case_min_max = (
        Min_Ind_Trxn_Amt is not None and Min_Ind_Trxn_Amt > 0 and
        Max_Ind_Trxn_Amt is not None and Max_Ind_Trxn_Amt > 0 and
        Total_Amount is None and
        N_transactions is not None and N_transactions > 0
    )
    case_total_only = (
        Total_Amount is not None and Total_Amount > 0 and
        Min_Ind_Trxn_Amt in [None, 0] and
        Max_Ind_Trxn_Amt in [None, 0] and
        N_transactions is not None and N_transactions > 0
    )
    case_total_and_bounds = (
        Total_Amount is not None and Total_Amount > 0 and
        Min_Ind_Trxn_Amt is not None and Min_Ind_Trxn_Amt > 0 and
        Max_Ind_Trxn_Amt is not None and Max_Ind_Trxn_Amt > 0
    )

    if case_total_and_bounds:
        N_min = math.ceil(Total_Amount / Max_Ind_Trxn_Amt)
        N_max = math.floor(Total_Amount / Min_Ind_Trxn_Amt)
        if N_min > N_max:
            raise ValueError("Incompatible Total_Amount with provided bounds")
        N_transactions = random.randint(N_min, N_max)
    if not (case_min_max or case_total_only or case_total_and_bounds):
        logger.info(
            "Warning: Invalid combination of inputs. "
            "Expected one of the following: "
            "a) N_transactions & Min_Ind_Trxn_Amt & Max_Ind_Trxn_Amt, "
            "b) N_transactions & Total_Amount (with Min_Ind_Trxn_Amt and Max_Ind_Trxn_Amt unset or zero), "
            "c) Total_Amount & Min_Ind_Trxn_Amt & Max_Ind_Trxn_Amt."
        )
        return {}
    
    trxns = {} #Dictionary to hold transactions
    trxn_channels = random.choices(Trxn_Channel, k = N_transactions)
    
    sample_deltas  =  random.choices(range((End_Date - Start_Date).days),k = N_transactions) #Get random number of days to be added to get new dates
    trxn_dates = [   Start_Date + timedelta(delta) for delta in sample_deltas] # TO DO: Add start and end date to the list
    #Convert back to string
    trxn_dates = [trxn_date.strftime("%Y-%m-%d") for trxn_date in trxn_dates]

    if case_min_max:
        trxn_amounts = np.round(
            np.random.uniform(low=Min_Ind_Trxn_Amt, high=Max_Ind_Trxn_Amt, size=N_transactions), 2
        )
    elif case_total_only:
        # Weights sampled near uniform to reduce variance
        epsilon = 1.0 / (2 * N_transactions)
        base = 1.0 / N_transactions
        weights = np.random.uniform(
            low=base - epsilon,
            high=base + epsilon,
            size=N_transactions
        )
        weights /= weights.sum()
        amounts = np.round(weights * Total_Amount, 2)
        diff = round(Total_Amount - amounts.sum(), 2)
        amounts[0] = round(amounts[0] + diff, 2)
        trxn_amounts = amounts
    else:  # case_total_and_bounds
        # Having estimated N_transactions, simply sample amounts uniformly within bounds
        trxn_amounts = np.round(
            np.random.uniform(
                low=Min_Ind_Trxn_Amt,
                high=Max_Ind_Trxn_Amt,
                size=N_transactions
            ),
            2
        )

    # Handle list of locations by sampling one per transaction
    if isinstance(Branch_or_ATM_Location, list):
        location_options = Branch_or_ATM_Location
    else:
        location_options = [Branch_or_ATM_Location]

    for i in range(N_transactions):
        trxns[(i+1)] = {"Originator_Name": Originator_Name , "Originator_Account_ID": Originator_Account_ID,"Originator_Customer_ID": Originator_Customer_ID,
                            "Beneficiary_Name": Beneficiary_Name, "Beneficiary_Account_ID": Beneficiary_Account_ID,"Beneficiary_Customer_ID": Beneficiary_Customer_ID,
                             "Trxn_Channel": trxn_channels[i], "Trxn_Date": trxn_dates[i], "Trxn_Amount":trxn_amounts[i],
                              "Branch_or_ATM_Location": random.choice(location_options) }

    return trxns

generate_transactions_schema = {
    "name": "generate_transactions",
    "description": "Generate a sequence of transactions based on the given parameters.",
    "parameters": {
        "type": "object",
        "properties": {
            "Originator_Name": {
                "type": "string",
                "description": "Entity or Customer originating the transactions"
            },
            "Originator_Account_ID": {
                "type": "string",
                "description": "Account of Entity or Customer originating the transactions"
            },
            "Originator_Customer_ID": {
                "type": "string",
                "description": "Customer ID of Entity or Customer originating the transactions"
            },
            "Beneficiary_Name": {
                "type": "string",
                "description": "Entity or Customer receiving the transactions"
            },
            "Beneficiary_Account_ID": {
                "type": "string",
                "description": "Account of Entity or Customer receiving the transactions"
            },
            "Beneficiary_Customer_ID": {
                "type": "string",
                "description": "Customer ID of Entity or Customer receiving the transactions"
            },
            "Trxn_Channel": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["Wire", "Cash", "Check", "Money Order"]
                },
                "description": "Transaction channels used for each transaction"
            },
            "Start_Date": {
                "type": "string",
                "format": "date",
                "description": "Date of the first transaction (YYYY-MM-DD)"
            },
            "End_Date": {
                "type": "string",
                "format": "date",
                "description": "Date of the last transaction (YYYY-MM-DD)"
            },
            "Min_Ind_Trxn_Amt": {
                "type": "number",
                "description": "Minimum individual transaction amount"
            },
            "Max_Ind_Trxn_Amt": {
                "type": "number",
                "description": "Maximum individual transaction amount"
            },
            "Branch_or_ATM_Location": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "Branch or ATM location(s) for transactions"
            },
            "N_transactions": {
                "type": "integer",
                "description": "Total number of transactions to generate"
            },
            "Total_Amount": {
                "type": "number",
                "description": "Total amount of all transactions"
            }
        },
        "required": [
            "Originator_Name",
            "Originator_Account_ID",
            "Originator_Customer_ID",
            "Beneficiary_Name",
            "Beneficiary_Account_ID",
            "Beneficiary_Customer_ID",
            "Trxn_Channel",
            "Start_Date",
            "End_Date",
            "Min_Ind_Trxn_Amt",
            "Max_Ind_Trxn_Amt",
            "Branch_or_ATM_Location"
        ],
        "additionalProperties": False
    }
}