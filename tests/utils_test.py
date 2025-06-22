import pandas as pd
from typing import Dict, Any
import unittest
import numpy as np

def assert_transaction_matches(test_case: unittest.TestCase, actual_df: pd.DataFrame, expected: Dict[str, Any]):
    test_case.assertGreater(len(actual_df), 0, "No transactions found.")

    total_amount = actual_df["Trxn_Amount"].sum()
    # Test Total Amount of trxns falls within range
    test_case.assertGreaterEqual(np.round(total_amount,2), expected["Min_Total_Amount"], "Total amount below expected range")
    test_case.assertLessEqual(np.round(total_amount,2), expected["Max_Total_Amount"], "Total amount above expected range")
    # Test Total No of trxns falls within range
    test_case.assertGreaterEqual(len(actual_df), expected["Min_N_trxns"], "Transaction count below expected range")
    test_case.assertLessEqual(len(actual_df), expected["Max_N_trxns"], "Transaction count above expected range")

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

    in_range_amt = actual_df["Trxn_Amount"].between(expected["Min_Ind_Amt"], expected["Max_Ind_Amt"], inclusive="both")
    test_case.assertGreaterEqual(in_range_amt.sum() / len(actual_df), 0.9,
                                 "Less than 90% transactions in amount range")

    actual_locs = set(actual_df["Branch_or_ATM_Location"].dropna().unique())
    expected_locs = set(expected["Branch_ATM_Location"])
    # Verify each actual location is one of the expected locations
    test_case.assertTrue(
        actual_locs.issubset(expected_locs),
        f"Unexpected transaction locations: {actual_locs - expected_locs}"
    )
