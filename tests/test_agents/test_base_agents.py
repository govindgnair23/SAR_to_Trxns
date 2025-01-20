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
        self.agent_config = get_agent_config(self.agent_configs, "Entity_Extraction_Agent")
        logging.info("Step 4: Extracted config for Entity Extraction  Agent correctly")
        self.entity_extraction_agent = instantiate_base_agent('Entity_Extraction_Agent',self.agent_config )
        logging.info("Step 5: Instantiated Entity Extraction Agent")
        self.message1 = """ 
                       John deposited $5000 in Cash into Acct #345723 at Bank of America. John sends $3000 to Jill's account at Chase.
                       Jill deposited $3000 in Cash into her Acct at Chase Bank.John and Jill own a business Acme Inc that has a Business account, Account #98765. 
                       John sends $2000 from Acct #345723 to Account #98765. Jill sends $1000 from her Acct at Chase Bank to Acct #98765.
                       """
        
        self.message2 = ""
        self.summary_prompt = self.agent_config.get("summary_prompt") 
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

class Test_Entity_Resolution_Agent(unittest.TestCase):
    '''
    Tests for the Entity Resolution Agent
    '''

    def setUp(self):
        self.agent_configs = load_agents_from_single_config('configs/agents_config.yaml')
        logging.info("Step 1: All agent configurations read")
        self.sar_agent_config = get_agent_config(self.agent_configs, "SAR_Agent")
        logging.info("Step 2: Extracted config for SAR Agent")
        self.sar_agent = instantiate_base_agent('SAR_Agent',self.sar_agent_config )
        logging.info("Step 3: Instantiated SAR Agent")
        self.agent_config = get_agent_config(self.agent_configs, "Entity_Resolution_Agent")
        logging.info("Step 4: Extracted config for Entity Resolution Agent correctly")
        self.entity_resolution_agent = instantiate_base_agent('Entity_Resolution_Agent',self.agent_config )
        logging.info("Step 5: Instantiated Entity Extraction Agent")
        self.message1 = """ 
                        1) Account_IDs = ["345723","98765","12345","99999","Dummy_Acct_1"]
      
                        2) Acct_to_Cust =  {"345723": "John, "99999":"John", "12345":"Jill", "Dummy_Acct_1" : "Jill","98765": "Acme Inc"}

                        3) Acct_to_FI =  {"345723":"Bank of America","99999":"Bank of America","12345":"Bank of America","Dummy_Acct_1":"Chase Bank", "98765":"Dummy_Bank_1" }

                        4) Narrative: </n>
                        John deposited $5000 each in Cash into Acct #345723 and Acct #99999, both of which are at Bank of America. John sends $4000  from Acct #345723 to Jill's account at  Chase. Jill deposited $3000 in Cash into her Acct at Chase Bank and wired $2000 to her Acct #12345 at Bank of America .John and Jill own a business Acme Inc that has a  Business account, Account #98765 . John sends $2000 from Acct #99999 to Account #98765. Jill sends $1000 from her Acct at Chase Bank to Acct #98765.
                       """
        
    
        self.summary_prompt = self.agent_config.get("summary_prompt") 
        logging.info("Step 5: Read summary prompt for Entity Extraction Agent")         

        self.results_dict1 = create_two_agent_chat(self.sar_agent,self.entity_resolution_agent,self.message1,self.summary_prompt)
        #self.results_dict2 = create_two_agent_chat(self.sar_agent,self.entity_extraction_agent,self.message2,self.summary_prompt)

    def test_base_case(self):
        """
        Validate the base-case scenario from setUp (message1).
        Checks that:
          - The output dictionary contains the expected FIs.
          - Each account is assigned correctly to its FI.
          - Multiple accounts owned by the same customer in one FI share the same CUST ID.
        """
        results = self.results_dict1
        
        # 1) The final result should contain these three institutions
        self.assertIn("Bank of America", results, "Expected 'Bank of America' in the result.")
        self.assertIn("Chase Bank", results, "Expected 'Chase Bank' in the result.")
        self.assertIn("Dummy_Bank_1", results, "Expected 'Dummy_Bank_1' in the result.")

        # 2) Check accounts for Bank of America
        boa_mapping = results["Bank of America"]
        self.assertCountEqual(
            boa_mapping.keys(), 
            ["345723", "99999", "12345"],
            "Bank of America should have accounts 345723, 99999, and 12345."
        )
        
        # 3) Check Chase Bank
        chase_mapping = results["Chase Bank"]
        self.assertCountEqual(
            chase_mapping.keys(), 
            ["Dummy_Acct_1"],
            "Chase Bank should only have account Dummy_Acct_1."
        )
        
        # 4) Check Dummy_Bank_1
        dummy_mapping = results["Dummy_Bank_1"]
        self.assertCountEqual(
            dummy_mapping.keys(),
            ["98765"],
            "Dummy_Bank_1 should only have account 98765."
        )

        # 5) Validate that John (owner of 345723, 99999 at Bank of America) 
        #    has the same Customer ID for both accounts.
        self.assertEqual(
            boa_mapping["345723"], 
            boa_mapping["99999"], 
            "Both accounts owned by John must map to the same CUST ID at Bank of America."
        )

        # 6) Check that Jill’s account at Bank of America (12345) 
        #    does not share John’s ID
        self.assertNotEqual(
            boa_mapping["345723"], 
            boa_mapping["12345"],
            "Jill's CUST ID should differ from John's at Bank of America."
        )

        # 7) Optionally, ensure the IDs start with "CUST_"
        for fi_name, acct_map in results.items():
            for acct_id, cust_id in acct_map.items():
                self.assertTrue(cust_id.startswith("CUST_"), "Customer ID should begin with 'CUST_'.")
    
    def test_scenario_multi_owners_multi_fis(self):
        """
        Scenario: Multiple owners across multiple FIs. 
        """
        message = """ 
                   1) Account_IDs = ["C1","C2","C3","C4","C5"]
                   2) Acct_to_Cust = {"C1":"Fred","C2":"Gina","C3":"Fred","C4":"HedgeFund LLC","C5":"Gina"}
                   3) Acct_to_FI =  {"C1":"AlphaBank","C2":"AlphaBank","C3":"BetaBank","C4":"BetaBank","C5":"BetaBank"}
                   4) Narrative:
                      Fred has Acct #C1 at AlphaBank and Acct #C3 at BetaBank.
                      Gina has Acct #C2 at AlphaBank and Acct #C5 at BetaBank.
                      HedgeFund LLC has Acct #C4 at BetaBank.
                   """
        results = create_two_agent_chat(
            self.sar_agent,
            self.entity_resolution_agent,
            message,
            self.summary_prompt
        )
        # Check presence of FIs
        self.assertIn("AlphaBank", results)
        self.assertIn("BetaBank", results)
        
        # AlphaBank should have C1, C2 => Fred, Gina
        alpha_accts = list(results["AlphaBank"].keys())
        self.assertCountEqual(alpha_accts, ["C1", "C2"], "AlphaBank should only have accounts C1 and C2.")
        alpha_ids = set(results["AlphaBank"].values())
        self.assertEqual(len(alpha_ids), 2, "Expected two unique customer IDs (Fred, Gina) at AlphaBank.")

        # BetaBank should have C3, C4, C5 => Fred, HedgeFund LLC, Gina
        beta_accts = list(results["BetaBank"].keys())
        self.assertCountEqual(beta_accts, ["C3", "C4", "C5"], "BetaBank should have accounts C3, C4, and C5.")
        beta_ids = set(results["BetaBank"].values())
        self.assertEqual(len(beta_ids), 3, "Expected three unique customer IDs at BetaBank.")


if __name__ == '__main__':
    unittest.main()



