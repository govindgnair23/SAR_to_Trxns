import unittest
import logging
import os
import sys
from contextlib import redirect_stdout
from unittest.mock import patch
import main  
from agents.workflows import run_agentic_workflow1, run_agentic_workflow2


logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')



class TestWorkflow1(unittest.TestCase):
    '''
    test first workflow starting with the Entity Extraction Agent and ending with the Narrative Extraction Agent.
    '''
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.expected_dictionary = {
            'Entities': {
                'Individuals': ['John', 'Jill'],
                'Organizations': ['Acme Inc'],
                'Financial_Institutions': ['Bank of America', 'Chase Bank']
            },
            'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'],
            'Acct_to_FI': {
                '345723': 'Bank of America',
                '98765': 'Dummy_Bank_1',
                'Dummy_Acct_1': 'Chase Bank'
            },
            'Acct_to_Cust': {
                '345723': 'John',
                '98765': 'Acme Inc',
                'Dummy_Acct_1': 'Jill'
            },
            'FI_to_Acct_to_Cust': {
                'Bank of America': {'345723': 'CUST_001'},
                'Chase Bank': {'Dummy_Acct_1': 'CUST_002'},
                'Dummy_Bank_1': {'98765': 'CUST_003'}
            },
            'Narratives': {
                '345723': ("John deposited $5000 in Cash into Acct #345723 at Bank of America. "
                           "John sends $3000 to Jill's account at Chase."),
                '98765': ("John sends $2000 from Acct #345723 to Account #98765. "
                          "Jill sends $1000 from her Acct at Chase Bank to Acct #98765."),
                'Dummy_Acct_1': "Jill deposited $3000 in Cash into her Acct at Chase Bank."
            }
        }

        test_sar1  = '''
                John deposited $5000 in Cash into Acct #345723 at Bank of America. John sends $3000 to Jill's account at Chase.
                Jill deposited $3000 in Cash into her Acct at Chase Bank.John and Jill own a business Acme Inc that has a Business account, Account #98765. 
                John sends $2000 from Acct #345723 to Account #98765. Jill sends $1000 from her Acct at Chase Bank to Acct #98765.

            '''

        
 
        config_file = 'configs/agents_config.yaml' 
        cls.result = run_agentic_workflow1(test_sar1,config_file)

    def test_result_is_dict(self):
        """
        Test that the output from the workflow is a valid Python dictionary.
        """
        self.assertIsInstance(self.result, dict, "Expected result to be a dictionary, but got a different type.")



    def test_keys_in_dictionary(self):
        """
        Test that all expected top-level keys are present.
        """
        expected_keys = set(self.expected_dictionary.keys())
        result_keys = set(self.result.keys())

        self.assertSetEqual(
            result_keys,
            expected_keys,
            msg=f"Expected keys {expected_keys} but got {result_keys}"
        )

    def test_values_for_Entities(self):
        """
        Test that the 'Entities' key contains the correct sub-keys and values.
        """
        self.assertIn('Entities', self.result, "'Entities' key is missing in the result")

        expected_entities = self.expected_dictionary['Entities']
        actual_entities = self.result['Entities']

        # Check that all expected sub-keys under 'Entities' exist
        expected_entity_keys = set(expected_entities.keys())
        actual_entity_keys = set(actual_entities.keys())

        self.assertSetEqual(
            actual_entity_keys, 
            expected_entity_keys,
            msg=(f"Under 'Entities', expected keys {expected_entity_keys} "
                 f"but got {actual_entity_keys}")
        )

        # Check the lists of Individuals, Organizations, and Financial_Institutions
        self.assertEqual(
            actual_entities['Individuals'],
            expected_entities['Individuals'],
            msg=("Expected Individuals "
                 f"{expected_entities['Individuals']} "
                 f"but got {actual_entities['Individuals']}")
        )
        self.assertEqual(
            actual_entities['Organizations'],
            expected_entities['Organizations'],
            msg=("Expected Organizations "
                 f"{expected_entities['Organizations']} "
                 f"but got {actual_entities['Organizations']}")
        )
        self.assertEqual(
            actual_entities['Financial_Institutions'],
            expected_entities['Financial_Institutions'],
            msg=("Expected Financial_Institutions "
                 f"{expected_entities['Financial_Institutions']} "
                 f"but got {actual_entities['Financial_Institutions']}")
        )

    def test_values_for_Account_IDs(self):
        """
        Test that 'Account_IDs' is a list and has the correct values.
        """
        self.assertIn('Account_IDs', self.result, "'Account_IDs' key is missing in the result")

        expected_acct_ids = self.expected_dictionary['Account_IDs']
        actual_acct_ids = self.result['Account_IDs']

        self.assertEqual(
            actual_acct_ids, 
            expected_acct_ids,
            msg=f"Expected Account_IDs {expected_acct_ids} but got {actual_acct_ids}"
        )

    def test_values_for_Acct_to_FI(self):
        """
        Test the correctness of 'Acct_to_FI'.
        """
        self.assertIn('Acct_to_FI', self.result, "'Acct_to_FI' key is missing in the result")

        expected_acct_to_fi = self.expected_dictionary['Acct_to_FI']
        actual_acct_to_fi = self.result['Acct_to_FI']

        self.assertEqual(
            actual_acct_to_fi, 
            expected_acct_to_fi,
            msg=f"Expected Acct_to_FI {expected_acct_to_fi} but got {actual_acct_to_fi}"
        )

    def test_values_for_Acct_to_Cust(self):
        """
        Test the correctness of 'Acct_to_Cust'.
        """
        self.assertIn('Acct_to_Cust', self.result, "'Acct_to_Cust' key is missing in the result")

        expected_acct_to_cust = self.expected_dictionary['Acct_to_Cust']
        actual_acct_to_cust = self.result['Acct_to_Cust']

        self.assertEqual(
            actual_acct_to_cust, 
            expected_acct_to_cust,
            msg=f"Expected Acct_to_Cust {expected_acct_to_cust} but got {actual_acct_to_cust}"
        )

    def test_values_for_FI_to_Acct_to_Cust(self):
        """
        Test the correctness of 'FI_to_Acct_to_Cust'.
        """
        self.assertIn('FI_to_Acct_to_Cust', self.result, "'FI_to_Acct_to_Cust' key is missing in the result")

        expected_fi_acct_cust = self.expected_dictionary['FI_to_Acct_to_Cust']
        actual_fi_acct_cust = self.result['FI_to_Acct_to_Cust']

        self.assertEqual(
            actual_fi_acct_cust, 
            expected_fi_acct_cust,
            msg=(f"Expected FI_to_Acct_to_Cust {expected_fi_acct_cust} "
                 f"but got {actual_fi_acct_cust}")
        )

    def test_values_for_Narratives(self):
        """
        Test the correctness of 'Narratives'.
        """
        self.assertIn('Narratives', self.result, "'Narratives' key is missing in the result")

        expected_narratives = self.expected_dictionary['Narratives']
        actual_narratives = self.result['Narratives']

        self.assertEqual(
            actual_narratives, 
            expected_narratives,
            msg=f"Expected Narratives {expected_narratives} but got {actual_narratives}"
        )

class TestWorkflow2(unittest.TestCase):
    '''
    Testing workflow to take extracted narrative and return transactions.
    '''
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.expected_transactions_case1 = {
                    "345723": {
                                1: {
                                    "Originator_Name": "John",
                                    "Originator_Account_ID": "345723",
                                    "Originator_Customer_ID": "CUST_001",
                                    "Beneficiary_Name": "John",
                                    "Beneficiary_Account_ID": "345723",
                                    "Beneficiary_Customer_ID": "CUST_001",
                                    "Trxn_Channel": "Cash",
                                    "Trxn_Date": "2025-01-05",
                                    "Trxn_Amount": 5000,
                                    "Branch_or_ATM_Location": "Bank of America"
                                },
                                2: {
                                    "Originator_Name": "John",
                                    "Originator_Account_ID": "345723",
                                    "Originator_Customer_ID": "CUST_001",
                                    "Beneficiary_Name": "Jill",
                                    "Beneficiary_Account_ID": "Dummy_Acct_1",
                                    "Beneficiary_Customer_ID": "CUST_002",
                                    "Trxn_Channel": "Wire",
                                    "Trxn_Date": "2025-02-01",
                                    "Trxn_Amount": 3000,
                                    "Branch_or_ATM_Location": ""
                                }
                            }
                                    }
        cls.expected_trxns_stats_case2 = {
        } 
        cls.expected_trxns_stats_case3 = {
        } 


        test_input1  = {'Entities': 
                            {'Individuals': ['John', 'Jill'], 
                            'Organizations': ['Acme Inc'], 
                            'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                      'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                      'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                      'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                      'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                      'Narratives': {'345723': "John deposited $5000 in Cash into Acct #345723 at Bank of America on Jan 5,2025. John sends $3000 to Jill's account at Chase on Feb 1,2025."}}
        
        test_input2  = {'Entities': 
                            {'Individuals': ['John', 'Jill'], 
                            'Organizations': ['Acme Inc'], 
                            'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                      'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                      'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                      'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                      'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                      'Narratives': {'345723': "John sent a total of 25 Wires from Acct #345723 at Bank of America  to Jill's account at Chase. The Wire trxns occured between Jan 1,2025 and Jan 31,2025 \
                                      and ranged between $5000 and $10,000 "}}
        
        
        test_input3  = {'Entities': 
                            {'Individuals': ['John', 'Jill'], 
                            'Organizations': ['Acme Inc'], 
                            'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                      'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                      'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                      'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                      'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                      'Narratives': {'345723': "John sent a total of $500,000 from Acct #345723 at Bank of America  to Jill's account at Chase. The Wire trxns occured between Jan 1,2025 and Jan 31,2025 \
                                      and ranged between $5000 and $10,000 "}}
        
        test_input4  = {'Entities': 
                        {'Individuals': ['John', 'Jill'], 
                        'Organizations': ['Acme Inc'], 
                        'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                    'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                    'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                    'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                    'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                    'Narratives': {'345723': "John sent a total of $500,000 from Acct #345723 at Bank of America  to Jill's account at Chase. The 15 Wire trxns occured between Jan 1,2025 and Jan 31,2025 "}}

 
        config_file = 'configs/agents_config.yaml' 
        cls.result1 = run_agentic_workflow2(test_input1,config_file)

if __name__ == '__main__':
    unittest.main()

