import unittest
from utils import  load_agents_from_single_config, get_agent_config
from agents.agents import instantiate_base_agent , create_two_agent_chat
from autogen import initiate_chats
import json



class Test_Entity_Extraction_Agent(unittest.TestCase):
    '''
    Tests for the Entity Extraction Agent
    '''

    def setUp(self):
        self.agent_configs = load_agents_from_single_config('configs/agents_config.yaml')
        self.sar_agent_config = get_agent_config(self.agent_configs, "SAR_Agent")
        self.sar_agent = instantiate_base_agent('SAR_Agent',self.sar_agent_config )
        self.entity_extraction_agent_config = get_agent_config(self.agent_configs, "Entity_Extraction_Agent")
        self.entity_extraction_agent = instantiate_base_agent('Entity_Extraction_Agent',self.entity_extraction_agent_config )
        self.message1 = """ 
                       John deposited $5000 in Cash into Acct #345723 at Bank of America. John sends $3000 to Jill's account at Chase.
                       Jill deposited $3000 in Cash into her Acct at Chase Bank.John and Jill own a business Acme Inc that has a Business account, Account #98765. 
                       John sends $2000 from Acct #345723 to Account #98765. Jill sends $1000 from her Acct at Chase Bank to Acct #98765.
                       """
        
        self.message2 = ""
        self.summary_prompt = self.entity_extraction_agent_config.get("summary_prompt")          

        self.results_dict1 = create_two_agent_chat(self.sar_agent,self.entity_extraction_agent,self.message1,self.summary_prompt)
        self.results_dict2 = create_two_agent_chat(self.sar_agent,self.entity_extraction_agent,self.message2,self.summary_prompt)

    def test_output_elements(self):
        '''
        Test  all output elements have been extracted.
        '''
        keys = list(self.results_dict.keys())
        self.assertIn("Entities", keys, "Entities extracted")
        self.assertIn("Account_IDs", keys, "Account IDs extracted")
        self.assertIn("FI_to_Acct", keys, " Mapping from Financial institutions to Accounts IDs extracted")
        self.assertIn("Acct_to_Cust", keys, " Mapping from  Accounts IDs to Customers extracted")
        



    def test_entities(self): 
        '''
        Test All Entities are correctly extracted.
        '''
        expected_entities = {
        "Individuals": ["John", "Jill"],
        "Organizations": ["Acme Inc"],
        "Financial_Institutions": ["Bank of America", "Chase Bank"]
        }

        self.assertEqual(expected_entities["Individuals"],self.results_dict1["Individuals"], "Individuals have been correctly extracted as a list")
        self.assertEqual(expected_entities["Organizations"],self.results_dict1["Organizations"], "Organizations have been correctly extracted as a list")
        self.assertEqual(expected_entities["Financial_Institutions"],self.results_dict1["Financial_Institutions"], "Financial_Institutions have been correctly extracted as a list")

    def test_account_ids(self):
        '''
        Test All Account IDs are correctly extracted.
        '''
        expected_account_ids = {"345723", "98765"}
        expected_account_ids_dummy = {"Dummy_Acct_1"}
        self.assertTrue(expected_account_ids.issubset(self.results_dict1["Account_IDs"]), " Non Dummy Account IDs have been correctly extracted ")
        self.assertTrue(expected_account_ids_dummy.issubset(self.results_dict1["Account_IDs"]), " Dummy Account IDs have been correctly extracted ")

    def test_map_financial_institutions_to_accounts(self):
        '''
        Test that the agent correctly maps account IDs to financial institutions.

        '''
        expected_fis_to_accts = {
            "Bank of America": ["345723"],
            "Chase Bank": ["Dummy_Acct_1"],
            "Dummy_Bank_1": ["98765"]
        }

        self.assertEqual(expected_fis_to_accts["Bank of America"],self.results_dict1["FIs_to_Accts"]["Bank of America"], "Account IDs correctly extracted for Bank of America")
        self.assertEqual(expected_fis_to_accts["Chase Bank"],self.results_dict1["FIs_to_Accts"]["Chase Bank"], "Account IDs correctly extracted for Chase Bank")
        self.assertEqual(expected_fis_to_accts["Dummy_Bank_1"],self.results_dict1["FIs_to_Accts"]["Dummy_Bank_1"], "Account IDs correctly extracted for Dummy_Bank_1")



    def test_map_accounts_to_customers(self):
        """
        Test that the agent correctly maps account IDs to the corresponding customers.
        """
        expected_accts_to_customers = {
            "345723" : "John",
            "Dummy_Acct_1": "Jill",
            "98765": "Acme Inc"
        }

        self.assertEqual(expected_accts_to_customers["345723"],self.results_dict1["Acct_to_Cust"]["345723"], "Account IDs #345723 mapped correctly")
        self.assertEqual(expected_accts_to_customers["Dummy_Acct_1"],self.results_dict1["Acct_to_Cust"]["Dummy_Acct_1"], "Dummy_Acct_1 mapped correctly")
        self.assertEqual(expected_accts_to_customers["98765"],self.results_dict1["Acct_to_Cust"]["98765"], "Account IDs #98765 mapped correctly")

    def test_empty_narrative_handling(self):
        """
        Test that the agent responds appropriately to an empty narrative input.
        """
        self.assertEqual({},self.results_dict2["Individuals"], "Individual Entities are empty as expected for empty input")
        self.assertEqual({},self.results_dict2["Organizations"], "Organizations are empty as expected for empty input")
        self.assertEqual({},self.results_dict2["Financial_Institutions"], "Financial_Institutions are empty as expected for empty input")
        self.assertEqual([],self.results_dict2["Account_IDs"], "Account IDs are empty as expected for empty input ")
        self.assertEqual({},self.results_dict2["FIs_to_Accts"], " mapping between FIs to Accts are empty as expected for empty input ")
        self.assertEqual({},self.results_dict2["Acct_to_Cust"], " mapping between Acct to Cust are empty as expected for empty input ")
        

if __name__ == '__main__':
    unittest.main()



