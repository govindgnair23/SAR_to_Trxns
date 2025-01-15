import unittest
from utils import  load_agents_from_single_config, get_agent_config
from agents.agents import instantiate_base_agent , create_two_agent_chat
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')


class Test_Entity_Extraction_Agent(unittest.TestCase):
    '''
    Tests for the Entity Extraction Agent
    '''

    def setUp(self):
        self.agent_configs = load_agents_from_single_config('configs/agents_config.yaml')
        logging.info("Step 1: All agent configurations read")
        self.sar_agent_config = get_agent_config(self.agent_configs, "SAR_Agent")
        logging.info("Step 2: Extracted config for SAR Agent")
        self.sar_agent = instantiate_base_agent('SAR_Agent',self.sar_agent_config )
        logging.info("Step 3: Instantiated SAR Agent")
        self.entity_extraction_agent_config = get_agent_config(self.agent_configs, "Entity_Extraction_Agent")
        logging.info("Step 4: Extracted config for Entity Extraction  Agent correct;y")
        self.entity_extraction_agent = instantiate_base_agent('Entity_Extraction_Agent',self.entity_extraction_agent_config )
        logging.info("Step 5: Instantiated Entity Extraction Agent")
        self.message1 = """ 
                       John deposited $5000 in Cash into Acct #345723 at Bank of America. John sends $3000 to Jill's account at Chase.
                       Jill deposited $3000 in Cash into her Acct at Chase Bank.John and Jill own a business Acme Inc that has a Business account, Account #98765. 
                       John sends $2000 from Acct #345723 to Account #98765. Jill sends $1000 from her Acct at Chase Bank to Acct #98765.
                       """
        
        self.message2 = ""
        self.summary_prompt = self.entity_extraction_agent_config.get("summary_prompt") 
        logging.info("Step 5: Read summary prompt for Entity Extraction Agent")         

        self.results_dict1 = create_two_agent_chat(self.sar_agent,self.entity_extraction_agent,self.message1,self.summary_prompt)
        #self.results_dict2 = create_two_agent_chat(self.sar_agent,self.entity_extraction_agent,self.message2,self.summary_prompt)

    def test_output_elements(self):
        '''
        Test  all output elements have been extracted.
        '''
        keys = list(self.results_dict1.keys())
        self.assertIn("Entities", keys, "Entities not extracted")
        self.assertIn("Account_IDs", keys, "Account IDs not extracted")
        self.assertIn("Acct_to_FI", keys, " Mapping from Financial institutions to Accounts IDs not extracted")
        self.assertIn("Acct_to_Cust", keys, " Mapping from  Accounts IDs to Customers not extracted")
        



    def test_entities(self): 
        '''
        Test All Entities are correctly extracted.
        '''
        expected_entities = {
        "Individuals": ["John", "Jill"],
        "Organizations": ["Acme Inc"],
        "Financial_Institutions": ["Bank of America", "Chase Bank"]
        }

        self.assertEqual(expected_entities["Individuals"],self.results_dict1["Entities"]["Individuals"], "Individuals have  NOT been correctly extracted as a list")
        self.assertEqual(expected_entities["Organizations"],self.results_dict1["Entities"]["Organizations"], "Organizations have NOT been correctly extracted as a list")
        self.assertEqual(expected_entities["Financial_Institutions"],self.results_dict1["Entities"]["Financial_Institutions"], "Financial_Institutions have NOT been correctly extracted as a list")

    def test_account_ids(self):
        '''
        Test All Account IDs are correctly extracted.
        '''
        expected_account_ids = {"345723", "98765"}
        expected_account_ids_dummy = {"Dummy_Acct_1"}
        self.assertTrue(expected_account_ids.issubset(self.results_dict1["Account_IDs"]), " Non Dummy Account IDs have NOT been correctly extracted ")
        self.assertTrue(expected_account_ids_dummy.issubset(self.results_dict1["Account_IDs"]), " Dummy Account IDs have NOT been correctly extracted ")

    def test_map_accounts_to_financial_institutions(self):
        '''
        Test that the agent correctly maps account IDs to financial institutions.

        '''
       
        expected_acct_to_fi = {
            "345723": "Bank of America",
            "Dummy_Acct_1": "Chase Bank",
            "98765": "Dummy_Bank_1"
        }


        self.assertEqual(expected_acct_to_fi["345723"],self.results_dict1["Acct_to_FI"]["345723"], "Account ID  345723 NOT correctly mapped ")
        self.assertEqual(expected_acct_to_fi["Dummy_Acct_1"],self.results_dict1["Acct_to_FI"]["Dummy_Acct_1"], "Dummy_Acct_1 NOT correctly mapped")
        self.assertEqual(expected_acct_to_fi["98765"],self.results_dict1["Acct_to_FI"]["98765"], "Account ID  98765 NOT correctly mapped")



    def test_map_accounts_to_customers(self):
        """
        Test that the agent correctly maps account IDs to the corresponding customers.
        """
        expected_accts_to_customers = {
            "345723" : "John",
            "Dummy_Acct_1": "Jill",
            "98765": "Acme Inc"
        }

        self.assertEqual(expected_accts_to_customers["345723"],self.results_dict1["Acct_to_Cust"]["345723"], "Account IDs #345723 NOT mapped correctly")
        self.assertEqual(expected_accts_to_customers["Dummy_Acct_1"],self.results_dict1["Acct_to_Cust"]["Dummy_Acct_1"], "Dummy_Acct_1 NOT mapped correctly")
        self.assertEqual(expected_accts_to_customers["98765"],self.results_dict1["Acct_to_Cust"]["98765"], "Account IDs #98765 NOT mapped correctly")

   
if __name__ == '__main__':
    unittest.main()



