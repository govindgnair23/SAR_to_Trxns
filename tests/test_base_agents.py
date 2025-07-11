import unittest
from utils import  load_agents_from_single_config, get_agent_config, approximate_match_ratio
from agents.agents import instantiate_base_agent , create_two_agent_chat, instantiate_agents_for_trxn_generation
import logging
from agents.agent_utils import route


logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')



class Test_Entity_Extraction_Agent(unittest.TestCase):
    '''
    Tests for the Entity Extraction Agent
    '''

    @classmethod
    def setUpClass(cls):
        super(Test_Entity_Extraction_Agent, cls).setUpClass()
        cls.agent_configs = load_agents_from_single_config('configs/agents_config.yaml')
        logging.info("Step 1: All agent configurations read")
        cls.sar_agent_config = get_agent_config(cls.agent_configs, "SAR_Agent")
        logging.info("Step 2: Extracted config for SAR Agent")
        cls.sar_agent = instantiate_base_agent('SAR_Agent', cls.sar_agent_config)
        logging.info("Step 3: Instantiated SAR Agent")
        cls.agent_config = get_agent_config(cls.agent_configs, "Entity_Extraction_Agent")
        logging.info("Step 4: Extracted config for Entity Extraction  Agent correctly")
        cls.entity_extraction_agent = instantiate_base_agent('Entity_Extraction_Agent', cls.agent_config)
        logging.info("Step 5: Instantiated Entity Extraction Agent")
        cls.message1 = """ 
                       John deposited $5000 in Cash into Acct #345723 at Bank of America. John sends $3000 to Jill's account at Chase.
                       Jill deposited $3000 in Cash into her Acct at Chase Bank.John and Jill own a business Acme Inc that has a Business account, Account #98765. 
                       John sends $2000 from Acct #345723 to Account #98765. Jill sends $1000 from her Acct at Chase Bank to Acct #98765.
                       """
        cls.message2 = ""
        cls.summary_prompt = cls.agent_config.get("summary_prompt") 
        logging.info("Step 6: Read summary prompt for Entity Extraction Agent")         
        cls.results_dict1 = create_two_agent_chat(cls.sar_agent, cls.entity_extraction_agent, cls.message1, cls.summary_prompt)


    def setUp(self):
        logging.info(f"Running {self.__class__.__name__}.{self._testMethodName}")


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

    @classmethod
    def setUpClass(cls):
        super(Test_Entity_Resolution_Agent, cls).setUpClass()
        cls.agent_configs = load_agents_from_single_config('configs/agents_config.yaml')
        logging.info("Step 1: All agent configurations read")
        cls.sar_agent_config = get_agent_config(cls.agent_configs, "SAR_Agent")
        logging.info("Step 2: Extracted config for SAR Agent")
        cls.sar_agent = instantiate_base_agent('SAR_Agent', cls.sar_agent_config)
        logging.info("Step 3: Instantiated SAR Agent")
        cls.agent_config = get_agent_config(cls.agent_configs, "Entity_Resolution_Agent")
        logging.info("Step 4: Extracted config for Entity Resolution Agent correctly")
        cls.entity_resolution_agent = instantiate_base_agent('Entity_Resolution_Agent', cls.agent_config)
        logging.info("Step 5: Instantiated Entity Resolution Agent")
        cls.message1 = """ 
                        1) Account_IDs = ["345723","98765","12345","99999","Dummy_Acct_1"]
      
                        2) Acct_to_Cust =  {"345723": "John", "99999":"John", "12345":"Jill", "Dummy_Acct_1" : "Jill", "98765": "Acme Inc"}

                        3) Acct_to_FI =  {"345723":"Bank of America","99999":"Bank of America","12345":"Bank of America","Dummy_Acct_1":"Chase Bank", "98765":"Dummy_Bank_1" }

                        4) Narrative: </n>
                        John deposited $5000 each in Cash into Acct #345723 and Acct #99999, both of which are at Bank of America. John sends $4000  from Acct #345723 to Jill's account at  Chase. Jill deposited $3000 in Cash into her Acct at Chase Bank and wired $2000 to her Acct #12345 at Bank of America. John and Jill own a business Acme Inc that has a  Business account, Account #98765. John sends $2000 from Acct #99999 to Account #98765. Jill sends $1000 from her Acct at Chase Bank to Acct #98765.
                        """
        cls.summary_prompt = cls.agent_config.get("summary_prompt") 
        logging.info("Step 6: Read summary prompt for Entity Resolution Agent")
        cls.results_dict1 = create_two_agent_chat(cls.sar_agent, cls.entity_resolution_agent, cls.message1, cls.summary_prompt)


    def setUp(self):
        logging.info(f"Running {self.__class__.__name__}.{self._testMethodName}")

    def test_base_case(self):
        """
        Validate the base-case scenario from setUp (message1).
        Checks that:
          - The output dictionary contains the expected FIs.
          - Each account is assigned correctly to its FI.
          - Multiple accounts owned by the same customer in one FI share the same CUST ID.
        """
        results = self.results_dict1["FI_to_Acct_to_Cust"]
        results_fis = results.keys()
        
        # 1) The final result should contain these three institutions
        self.assertIn("Bank of America", results_fis, "Expected 'Bank of America' in the result.")
        self.assertIn("Chase Bank", results_fis, "Expected 'Chase Bank' in the result.")
        self.assertIn("Dummy_Bank_1", results_fis, "Expected 'Dummy_Bank_1' in the result.")

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

        results_ = results["FI_to_Acct_to_Cust"]
        # Check presence of FIs
        self.assertIn("AlphaBank", results_)
        self.assertIn("BetaBank", results_)
        
        # AlphaBank should have C1, C2 => Fred, Gina
        alpha_accts = list(results_["AlphaBank"].keys())
        self.assertCountEqual(alpha_accts, ["C1", "C2"], "AlphaBank should only have accounts C1 and C2.")
        alpha_ids = set(results_["AlphaBank"].values())
        self.assertEqual(len(alpha_ids), 2, "Expected two unique customer IDs (Fred, Gina) at AlphaBank.")

        # BetaBank should have C3, C4, C5 => Fred, HedgeFund LLC, Gina
        beta_accts = list(results_["BetaBank"].keys())
        self.assertCountEqual(beta_accts, ["C3", "C4", "C5"], "BetaBank should have accounts C3, C4, and C5.")
        beta_ids = set(results_["BetaBank"].values())
        self.assertEqual(len(beta_ids), 3, "Expected three unique customer IDs at BetaBank.")


class Test_Narrative_Extraction_Agent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        logging.info("Loading agent configs...")
        cls.agent_configs = load_agents_from_single_config('configs/agents_config.yaml')

        logging.info("Step 1: All agent configurations read")
        cls.sar_agent_config = get_agent_config(cls.agent_configs, "SAR_Agent")
        logging.info("Step 2: Extracted config for SAR Agent")
        cls.sar_agent = instantiate_base_agent('SAR_Agent',cls.sar_agent_config )
        logging.info("Step 3: Instantiated SAR Agent")

        
        cls.agent_config = get_agent_config(cls.agent_configs, "Narrative_Extraction_Agent")
        logging.info("Step 4: Extracting Narrative_Extraction_Agent config...")

        
        cls.narrative_extraction_agent = instantiate_base_agent('Narrative_Extraction_Agent', 
                                                                 cls.agent_config)
        
        logging.info("Step 5: Instantiating Narrative Extraction Agent...")

        # Example message & expected result
        cls.message1 = """
        1) Account_IDs = ["345723","98765","12345","99999","Dummy_Acct_1"]
        2) Acct_to_Cust =  {"345723": "John","99999":"John","12345":"Jill","Dummy_Acct_1" : "Jill","98765": "Acme Inc"}
        3) Acct_to_FI =  {"345723":"Bank of America","99999":"Bank of America","12345":"Bank of America",
                          "Dummy_Acct_1":"Chase Bank","98765":"Dummy_Bank_1"}
        4) Narrative:
           John deposited $5000 each in Cash into Acct #345723 and Acct #99999, both of which are at Bank of America on Jan 1, 2025 . 
           John sends $4000  from Acct #345723 to Jill's account at Chase Bank on Jan 15,2025.
            Jill deposited $3000 in Cash into her Acct at Chase Bank on Jan 17,2025  and  
            then wired $2000 from that account to her Acct #12345 at Bank of America on Jan 19,2025 .John and Jill own a business Acme Inc that has a  Business account, Account #98765 . John sends $2000 from Acct #99999 to Account #98765 on Feb 1,2025. Jill sends $1000 from her Acct at Chase Bank to Acct #98765 by Wire on Feb 7,2025.
        """

        cls.expected_dict1 = {
            "345723": 
                { "Trxn_Set_1": "John deposited $5000 in Cash into Acct #345723 at Bank of America on Jan 1, 2025", 
                  "Trxn_Set_2": "John sends $4000  from Acct #345723 to Jill's account at Chase Bank on Jan 15,2025" },
            "98765":   {"Trxn_Set_1": " John sends $2000 from Acct #99999 to Account #98765 on Feb 1,2025",
                        "Trxn_Set_2": "Jill sends $1000 from her Acct at Chase Bank to Acct #98765 by Wire on Feb 7,2025"} ,
            "12345":  {"Trxn_Set_1": "Jill wired $2000 from her Acct at Chase Bank to her Acct #12345 at Bank of America on Jan 19,2025" },
            "99999":  {'Trxn_Set_1': 'John deposited $5000 each in Cash into Acct #99999 at Bank of America on Jan 1, 2025.',
                       'Trxn_Set_2': 'John sends $2000 from Acct #99999 to Account #98765 on Feb 1,2025.'},
            "Dummy_Acct_1": 
                    {"Trxn_Set_1": "John sends $4000  from Acct #345723 to Jill's account at  Chase Bank on Jan 15,2025",
                     "Trxn_Set_2": "Jill deposited $3000 in Cash into her Acct at Chase Bank on Jan 17,2025 " ,
                     "Trxn_Set_3": "Jill wired $2000 from her account at Chase Bank  to her Acct #12345 at Bank of America on Jan 19,2025"  ,
                     "Trxn_Set_4": "Jill sends $1000 from her Acct at Chase Bank to Acct #98765 by Wire on Feb 7,2025." 
                        }
        }

        
        cls.summary_prompt = cls.agent_config.get("summary_prompt") 
        logging.info("Step 6: Read summary prompt for narrative Extraction Agent")         

        results = create_two_agent_chat(cls.sar_agent,cls.narrative_extraction_agent,cls.message1,cls.summary_prompt)
        cls.results_dict1 = results["Narratives"]

    def setUp(self):
        logging.info(f"Running {self.__class__.__name__}.{self._testMethodName}")

    def test_output_format(self):
        """
        Validate the narrative extraction agent is outputting results in the right format
        """
        transaction_dict = self.results_dict1

        # The top-level keys we expect
        expected_keys = ["345723", "98765", "12345", "99999", "Dummy_Acct_1"]

        # Check that each expected key is present
        for key in expected_keys:
            self.assertIn(key, transaction_dict, f"Missing expected key: {key}")

        # Optionally, you may want to ensure the dictionary does not have extra keys
        self.assertEqual(set(transaction_dict.keys()), set(expected_keys))

        # Check each account's sub-dictionary
        for acct, sub_dict in transaction_dict.items():
            self.assertIsInstance(
                sub_dict, dict,
                f"The value for '{acct}' should be a dictionary, but got {type(sub_dict)}"
            )
            # Ensure at least one sub-key exists
            self.assertGreater(
                len(sub_dict), 0,
                f"No sub-entries found under account '{acct}'"
            )
            # Check that each sub-key follows a pattern like 'Trxn_set_#' or 'Trxn_Set_#'
            for sub_key in sub_dict.keys():
                self.assertRegex(
                    sub_key,
                    r'^Trxn_set_\d+$|^Trxn_Set_\d+$',
                    f"Sub-key '{sub_key}' under '{acct}' does not match expected pattern."
                )
        

    def test_narrative_match_approx(self):
        """
        Validate scenario 1 using approximate lexical match for the narrative text.
        The similarity ratio should exceed our threshold (e.g. 0.80).
        """


        # 1) Check that the same set of account IDs exist in both dicts
        self.assertEqual(
            set(self.results_dict1.keys()),
            set(self.expected_dict1.keys()),
            "Mismatch in the set of account IDs extracted."
        )

         # 2) Compare narratives account by account, sub-key by sub-key
        threshold = 0.80
        for acct_id, expected_sub_dict in self.expected_dict1.items():
            self.assertIn(
                acct_id,
                self.results_dict1,
                f"Account '{acct_id}' is missing in the agent's output."
            )
            actual_sub_dict = self.results_dict1[acct_id]

            # 2a) Ensure the set of transaction keys matches
            self.assertEqual(
                set(actual_sub_dict.keys()),
                set(expected_sub_dict.keys()),
                (
                    f"Mismatch in the sub-key set for account '{acct_id}'.\n"
                    f"Expected: {set(expected_sub_dict.keys())}\n"
                    f"Got:      {set(actual_sub_dict.keys())}"
                )
            )

            # 2b) Compare each transaction narrative using approximate matching
            for sub_key, expected_text in expected_sub_dict.items():
                actual_text = actual_sub_dict[sub_key].strip()
                ratio = approximate_match_ratio(expected_text.strip(), actual_text)
                
                self.assertTrue(
                    ratio >= threshold,
                    (
                        f"Transaction '{sub_key}' in account '{acct_id}' "
                        f"does not meet the similarity threshold of {threshold:.2f}.\n"
                        f"Got ratio={ratio:.2f}.\n"
                        f"Expected: {expected_text}\n"
                        f"Actual:   {actual_text}"
                    )
                )

    def test_narrative_contains_date_and_amount(self):
        """
        Ensure each extracted narrative contains at least one date and one transaction amount.
        """
        # Regex to match month-name dates or numeric dates
        date_pattern = r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{1,2},\s*\d{4}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b)'
        # Regex to match dollar amounts like $1,000 or $5000.00
        amount_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?'

        for acct_id, sub_dict in self.results_dict1.items():
            for tx_key, narrative in sub_dict.items():
                self.assertRegex(
                    narrative,
                    date_pattern,
                    f"Narrative '{tx_key}' for account '{acct_id}' does not contain a date."
                )
                self.assertRegex(
                    narrative,
                    amount_pattern,
                    f"Narrative '{tx_key}' for account '{acct_id}' does not contain an amount."
                )




class TestRouterAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Runs once before any test methods in this class are executed.
        Loads agent configs, instantiates agents, and runs the scenario so the results
        are available to all tests.
        """
        logging.info("Loading agent configs...")
        cls.agent_configs = load_agents_from_single_config('configs/agents_config.yaml')
    
        cls.agents = instantiate_agents_for_trxn_generation(cls.agent_configs)
        logging.info("All agents instantiated successfully")

        cls.router_agent = cls.agents["Router_Agent"]

    def setUp(self):
        logging.info(f"Running {self.__class__.__name__}.{self._testMethodName}")
        


    def test_correct_agent_invoked_case1(self):

        test_message1 =  {'Entities': 
                                {'Individuals': ['John', 'Jill'], 
                            'Organizations': ['Acme Inc'], 
                                'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                                'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                                'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                                'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                                'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                                'Narratives' : {"345723": 
                                        {
                                        "Trxn_Set_1":"John sent 25 wires to Acct #98765 between Jan 10,2025 and Feb 15, 2025. The trxns ranged from $1,000 to $5,000"} }
                                }
                            
        
        selected_agent = route(self.agents, narrative=test_message1)

        # Test that right agent is selected
        self.assertEqual(
            selected_agent,
            "Transaction_Generation_Agent_w_Tool",
            f"Expected 'Transaction_Generation_Agent_w_Tool' but got '{selected_agent}'"
        )

    def test_correct_agent_invoked_case2(self):

        test_message2 =  {'Entities': 
                                {'Individuals': ['John', 'Jill'], 
                            'Organizations': ['Acme Inc'], 
                                'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                                'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                                'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                                'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                                'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                                'Narratives' : {"345723": 
                                        {
                                        "Trxn_Set_1":"John sent 2 wires to Acct #98765 on Jan 10,2025 and Feb 15, 2025. The trxns were $4,400 and $6,5000"} }
                                }
                            
        selected_agent = route(self.agents, narrative=test_message2)
        # Test that right agent is selected
        self.assertEqual(
            selected_agent,
            "Transaction_Generation_Agent",
            f"Expected 'Transaction_Generation_Agent' but got '{selected_agent}'"
        )


    def test_correct_agent_invoked_case3(self):

        test_message3 =  {'Entities': 
                                {'Individuals': ['John', 'Jill'], 
                            'Organizations': ['Acme Inc'], 
                                'Financial_Institutions': ['Bank of America', 'Chase Bank']},
                                'Account_IDs': ['345723', '98765', 'Dummy_Acct_1'], 
                                'Acct_to_FI': {'345723': 'Bank of America', 'Dummy_Acct_1': 'Chase Bank', '98765': 'Dummy_Bank_1'},
                                'Acct_to_Cust': {'345723': 'John', 'Dummy_Acct_1': 'Jill', '98765': 'Acme Inc'}, 
                                'FI_to_Acct_to_Cust': {'Bank of America': {'345723': 'CUST_001'}, 'Chase Bank': {'Dummy_Acct_1': 'CUST_002'}, 'Dummy_Bank_1': {'98765': 'CUST_003'}},
                                'Narratives' : {"345723": 
                                        {
                                        "Trxn_Set_1":"Checks for $9,800 were issued and subsequently deposited at Bank of America on 06/04, 06/05, 06/10, 06/11, 06/12, and 06/13. The source of the cash is unknown, and this pattern appears to evade the reporting requirements of the Bank Secrecy Act."} }
                                }
                            
        selected_agent = route(self.agents, narrative=test_message3)
        # Test that right agent is selected
        self.assertEqual(
            selected_agent,
            "Transaction_Generation_Agent",
            f"Expected 'Transaction_Generation_Agent' but got '{selected_agent}'"
        )

       
        

if __name__ == "__main__":
    unittest.main(verbosity=2)



