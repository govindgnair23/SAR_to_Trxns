import unittest
import logging
from contextlib import redirect_stdout
from unittest.mock import patch
from agents.workflows import run_agentic_workflow1, run_agentic_workflow2
from utils import  compare_dicts, assert_dict_structure
import pandas as pd
from typing import Dict, Any
from tests.utils_test import assert_transaction_matches

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
                'Organizations': [],
                'Financial_Institutions': ['Bank of America', 'Chase Bank']
            },
            'Account_IDs': ['345723', '99999', 'Dummy_Acct_1'],
            'Acct_to_FI': {
                '345723': 'Bank of America',
                '99999': 'Bank of America',
                'Dummy_Acct_1': 'Chase Bank'
            },
            'Acct_to_Cust': {
                '345723': 'John',
                '99999': 'John',
                'Dummy_Acct_1': 'Jill'
            },
            'FI_to_Acct_to_Cust': {
                'Bank of America': {'345723': 'CUST_001','99999':'CUST_001' },
                'Chase Bank': {'Dummy_Acct_1': 'CUST_002'},
            },
            'Narratives' : {"345723": 
                                {"Trxn_Set_1":"John deposited $5000 in Cash into Acct #345723 at Bank of America on Jan 1,2025.", 
                                "Trxn_Set_2": "John sends $4000  from Acct #345723 to Jill's account at  Chase on Jan 15,2025" } ,
                            "Dummy_Acct_1": 
                              {"Trxn_Set_1": "John sends $4000  from Acct #345723 to Jill's account at Chase Bank on Jan 15,2025"} ,
                            "99999": 
                              {"Trxn_Set_1": "John deposited $5000  in Cash into  Acct #99999 at Bank of America on Jan 1, 2025" } 
        } }

        test_sar1  = '''
                 John deposited $5000 each in Cash into Acct #345723 and Acct #99999, both of which are at Bank of America on Jan 1, 2025 . 
                 John sends $4000  from Acct #345723 to Jill's account at Chase Bank on Jan 15,2025. 
                  

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

        
    def test_narratives_structure_dynamic(self):
        """
        Test the 'Narratives' dictionary has the right keys and sub-keys.
        """
        expected_narratives = self.expected_dictionary["Narratives"]
        actual_narratives = self.result["Narratives"]
        assert_dict_structure(self, expected_narratives, actual_narratives)


class TestWorkflow2(unittest.TestCase):
    '''
    Testing workflow to take extracted narrative and return transactions.
    '''
    @classmethod
    def setUpClass(cls):
        super().setUpClass()


        cls.test_input1  = {'Entities': 
                            {'Individuals': ['John', 'Jill'], 
                            'Organizations': ['Acme Inc'], 
                            'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                      'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                      'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                      'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                      'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                      'Narratives' : {"345723": 
                             {
                              "Trxn_Set_1":"John deposited $5000 in Cash into Acct #345723 at Bank of America on Jan 15,2025 at"
                              "the Manhattan Branch"} }
                     }
        
        cls.expected_results1 = {"Originator_Account_ID": "345723",
                                "Beneficiary_Account_ID": "345723",
                                "Total_Amount": 5000,
                                "Trxn_Type": ["Cash"],
                                "Min_Date": "2025-01-15",
                                "Max_Date": "2025-01-15",
                                "Branch_ATM_Location": ["Manhattan"],
                                "Min_Ind_Amt":5000,
                                "Max_Ind_Amt":5000,
                                "N_trxns": 1}
        
        cls.test_input2  = {'Entities': 
                            {'Individuals': ['John', 'Jill'], 
                            'Organizations': ['Acme Inc'], 
                            'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                      'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                      'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                      'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                      'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                      'Narratives' : {"345723": 
                             {
                              "Trxn_Set_1":"John sent a total of 25 Wires from Acct #345723 at Bank of America  to Jill's account at Chase. The Wire trxns occured between Jan 1,2025 and Jan 31,2025 and ranged between $5000 and $10,000 "} }
                     }
                    
            

        cls.expected_results2 = {"Originator_Account_ID": "345723",
                                "Beneficiary_Account_ID": "Dummy_Acct_1",
                                "Total_Amount": 25*7500,
                                "Trxn_Type": ["Wire"],
                                "Min_Date": "2025-01-01",
                                "Max_Date": "2025-01-31",
                                "Branch_ATM_Location": [''],
                                "Min_Ind_Amt":5000,
                                "Max_Ind_Amt":10000,
                                "N_trxns": 25}
        

        cls.test_input3  = {'Entities': 
                            {'Individuals': ['John', 'Jill'], 
                            'Organizations': ['Acme Inc'], 
                            'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                      'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                      'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                      'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                      'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                      'Narratives': {'345723': {"Trxn_Set_1":"John sent a total of $500,000 from Acct #345723 at Bank of America  to Jill's account at Chase. The Wire trxns occured between Jan 1,2025 and Jan 31,2025 \
                                      and ranged between $5000 and $10,000 "}}}
        
        cls.expected_results3 = {"Originator_Account_ID": "345723",
                                "Beneficiary_Account_ID": "Dummy_Acct_1",
                                "Total_Amount": 500000,
                                "Trxn_Type": ["Wire"],
                                "Min_Date": "2025-01-01",
                                "Max_Date": "2025-01-31",
                                "Branch_ATM_Location": [''],
                                "Min_Ind_Amt":5000,
                                "Max_Ind_Amt":10000,
                                "N_trxns": 500000/7500}
        
        cls.test_input4  = {'Entities': 
                        {'Individuals': ['John', 'Jill'], 
                        'Organizations': ['Acme Inc'], 
                        'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                    'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                    'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                    'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                    'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                    'Narratives': {'345723':{"Trxn_Set_1":"John sent a total of $500,000 from Acct #345723 at Bank of America  to Jill's account at Chase. The 15 Wire trxns occured between Jan 1,2025 and Jan 31,2025 "}}}

        cls.expected_results4 = {"Originator_Account_ID": "345723",
                                "Beneficiary_Account_ID": "Dummy_Acct_1",
                                "Total_Amount": 500000,
                                "Trxn_Type": ["Wire"],
                                "Min_Date": "2025-01-01",
                                "Max_Date": "2025-01-31",
                                "Branch_ATM_Location": [''],
                                "Min_Ind_Amt":500000/15,
                                "Max_Ind_Amt":499985,
                                "N_trxns": 15}

        cls.config_file = 'configs/agents_config.yaml' 
        #Use this to test outputs follow expected format.
        #cls.result1 = run_agentic_workflow2(cls.test_input1,cls.config_file)
        cls.result2 = run_agentic_workflow2(cls.test_input2,cls.config_file)
        # cls.result3 = run_agentic_workflow2(cls.test_input3,cls.config_file)
        # cls.result4 = run_agentic_workflow2(cls.test_input4,cls.config_file)

    def test_result_is_dataframe(self):
        """
        Test that the output from the workflow is a valid Python DataFrame.
        """
        # self.assertIsInstance(self.result1, pd.DataFrame, "Expected result to be a dictionary, but got a different type.")
        self.assertIsInstance(self.result2, pd.DataFrame, "Expected result to be a dictionary, but got a different type.")
        # self.assertIsInstance(self.result3, pd.DataFrame, "Expected result to be a dictionary, but got a different type.")
        # self.assertIsInstance(self.result4, pd.DataFrame, "Expected result to be a dictionary, but got a different type.")



    # def test_trxns_case1(self):
    #     assert_transaction_matches(self, self.result1, self.expected_results1)

    def test_trxns_case2(self):
        assert_transaction_matches(self, self.result2, self.expected_results2)
    
    # def test_trxns_case3(self):
    #     assert_transaction_matches(self, self.result3, self.expected_results3)
    
    # def test_trxns_case4(self):
    #     assert_transaction_matches(self, self.result4, self.expected_results4)

 