import pandas as pd
from typing import Dict, Any
import unittest

def assert_transaction_matches(test_case: unittest.TestCase, actual_df: pd.DataFrame, expected: Dict[str, Any]):
    test_case.assertGreater(len(actual_df), 0, "No transactions found.")

    total_amount = actual_df["Trxn_Amount"].sum()
    #Test Total Amount of trxns
    test_case.assertAlmostEqual(total_amount, expected["Total_Amount"], delta=0.1 * expected["Total_Amount"],
                                msg="Total amount mismatch")
    #Test Total No of Trxns
    test_case.assertEqual(len(actual_df), expected["N_trxns"], "Transaction count mismatch")

    test_case.assertTrue((actual_df["Originator_Account_ID"] == expected["Originator_Account_ID"]).all(),
                         "Mismatch in Originator_Account_ID")
    test_case.assertTrue((actual_df["Beneficiary_Account_ID"] == expected["Beneficiary_Account_ID"]).all(),
                         "Mismatch in Beneficiary_Account_ID")

    actual_channels = set(actual_df["Trxn_Channel"].dropna().unique())
    expected_channels = set(expected["Trxn_Type"])
    test_case.assertEqual(actual_channels, expected_channels, "Mismatch in transaction types/channels")

    min_date = pd.to_datetime(expected["Min_Date"])
    max_date = pd.to_datetime(expected["Max_Date"])
    actual_df["Trxn_Date"] = pd.to_datetime(actual_df["Trxn_Date"], errors="coerce")
    date_mask = (actual_df["Trxn_Date"] >= min_date) & (actual_df["Trxn_Date"] <= max_date)
    test_case.assertGreaterEqual(date_mask.sum() / len(actual_df), 0.9,
                                 "Less than 90% transactions in date range")

    in_range_amt = actual_df["Trxn_Amount"].between(expected["Min_Ind_Amt"], expected["Max_Ind_Amt"])
    test_case.assertGreaterEqual(in_range_amt.sum() / len(actual_df), 0.9,
                                 "Less than 90% transactions in amount range")

    actual_locs = set(actual_df["Branch_or_ATM_Location"].dropna().unique())
    expected_locs = set(expected["Branch_ATM_Location"])
    test_case.assertEqual(actual_locs, expected_locs, "Mismatch in transaction locations")