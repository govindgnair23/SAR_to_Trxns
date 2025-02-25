import unittest
from utils import  load_agents_from_single_config, get_agent_config, approximate_match_ratio
from agents.agents import instantiate_base_agent , create_two_agent_chat
import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')


## Todo : Need to change the setUp to setUp class. 

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


class Test_Narrative_Extraction_Agent(unittest.TestCase):
    
    def setUp(self):
        logging.info("Loading agent configs...")
        self.agent_configs = load_agents_from_single_config('configs/agents_config.yaml')

        logging.info("Step 1: All agent configurations read")
        self.sar_agent_config = get_agent_config(self.agent_configs, "SAR_Agent")
        logging.info("Step 2: Extracted config for SAR Agent")
        self.sar_agent = instantiate_base_agent('SAR_Agent',self.sar_agent_config )
        logging.info("Step 3: Instantiated SAR Agent")

        
        self.agent_config = get_agent_config(self.agent_configs, "Narrative_Extraction_Agent")
        logging.info("Step 4: Extracting Narrative_Extraction_Agent config...")

        
        self.narrative_extraction_agent = instantiate_base_agent('Narrative_Extraction_Agent', 
                                                                 self.agent_config)
        
        logging.info("Step 5: Instantiating Narrative Extraction Agent...")

        # Example message & expected result
        self.message1 = """
        1) Account_IDs = ["345723","98765","12345","99999","Dummy_Acct_1"]
        2) Acct_to_Cust =  {"345723": "John","99999":"John","12345":"Jill","Dummy_Acct_1" : "Jill","98765": "Acme Inc"}
        3) Acct_to_FI =  {"345723":"Bank of America","99999":"Bank of America","12345":"Bank of America",
                          "Dummy_Acct_1":"Chase Bank","98765":"Dummy_Bank_1"}
        4) Narrative:
           John deposited $5000 each in Cash into Acct #345723 and Acct #99999, both of which are at Bank of America.
           John sends $4000  from Acct #345723 to Jill's account at  Chase Bank.
           Jill deposited $3000 in Cash into her Acct at Chase Bank and then wired $2000 to her Acct #12345 at Bank of America.
           John and Jill own a business Acme Inc that has a Business account, Account #98765.
           John sends $2000 from Acct #99999 to Account #98765.
           Jill sends $1000 from her Acct at Chase Bank to Acct #98765 by Wire.
        """

        self.expected_dict1 = {
            "345723": (
                "John deposited $5000 each in Cash into Acct #345723 at Bank of America. "
                "John sends $4000  from Acct #345723 to Jill's account at  Chase."
            ),
            "98765":  "John sends $2000 from Acct #99999 to Account #98765.",
            "12345":  (
                "Jill deposited $3000 in Cash into her Acct at Chase Bank and then wired "
                "$2000 to her Acct #12345 at Bank of America"
            ),
            "99999":  "John sends $2000 from Acct #99999 to Account #98765",
            "Dummy_Acct_1": (
                "John sends $4000  from Acct #345723 to Jill's account at  Chase Bank. "
                "Jill deposited $3000 in Cash into her Acct at Chase Bank and  then  wired "
                "$2000 from that account to her Acct #12345 at Bank of America. "
                "Jill sends $1000 from her Acct at Chase Bank to Acct #98765 by Wire."
            )
        }

        
        self.summary_prompt = self.agent_config.get("summary_prompt") 
        logging.info("Step 6: Read summary prompt for narrative Extraction Agent")         

        self.results_dict1 = create_two_agent_chat(self.sar_agent,self.narrative_extraction_agent,self.message1,self.summary_prompt)
       

        

    def test_scenario_1_extraction_approx(self):
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

        # 2) Compare the narratives for each account using approximate matching
        threshold = 0.80
        for acct_id, expected_text in self.expected_dict1.items():
            self.assertIn(
                acct_id,
                self.results_dict1,
                f"Account {acct_id} is missing in the agent's output."
            )

            actual_text = self.results_dict1[acct_id].strip()
            ratio = approximate_match_ratio(expected_text.strip(), actual_text)
            
            self.assertTrue(
                ratio >= threshold,
                (
                    f"Narrative for account {acct_id} does not meet the similarity "
                    f"threshold of {threshold}. Got ratio={ratio:.2f}.\n"
                    f"Expected: {expected_text}\n"
                    f"Actual:   {actual_text}"
                )
            )


class Test_Transaction_Generation_Agent(unittest.TestCase):
    """
    Tests for the AI Agent that synthesizes transactions from a narrative.
    This agent:
      1) Reads the dictionary 'Narrative' with account IDs as keys and textual descriptions as values.
      2) Looks up account owners (individuals/organizations) and FIs.
      3) Extracts each described transaction.
      4) Returns a Python dictionary keyed by Transaction ID (e.g. 1, 2, 3).
    """

    
    def setUpClass(self):
        """
        Runs once before any test methods in this class are executed.
        Loads agent configs, instantiates agents, and runs the scenario so the results
        are available to all tests.
        """
        logging.info("Loading agent configs...")
        self.agent_configs = load_agents_from_single_config('configs/agents_config.yaml')

        logging.info("Step 1: All agent configurations read")
        self.sar_agent_config = get_agent_config(self.agent_configs, "SAR_Agent")
        logging.info("Step 2: Extracted config for SAR Agent")
        self.sar_agent = instantiate_base_agent('SAR_Agent', self.sar_agent_config)
        logging.info("Step 3: Instantiated SAR Agent")
        
        logging.info("Extracting Transaction_Generation_Agent config...")
        self.agent_config = get_agent_config(self.agent_configs, "Transaction_Generation_Agent")

        logging.info("Instantiating Transaction Generation Agent...")
        self.transaction_generation_agent = instantiate_base_agent(
            'Transaction_Generation_Agent', 
            self.agent_config
        )

        # Example test message we will reuse from instructions
        self.test_message = """
        1) Narrative = {
          "345723": "John deposited $5000 in Cash into Acct #345723 at the Main Road, NY Branch of Bank of America on Jan 4, 2024. \
                     John sends $3000 to Acme Inc's account at Bank of America by Wire on Jan 6, 2024. \
                     John wrote a check to Jill from Acct #345723 on Jan 8, 2024 for $1,000"
        }
        2) Acct_to_Cust = {"345723": "John", "Dummy_001":"Jill", "98765":"Acme Inc"}
        3) Acct_to_FI = {"345723":"Bank of America","98765":"Bank of America", "Dummy_001":"Chase Bank"}
        4) FI_to_Acct_to_Cust = {
             "Bank of America": {"345723": "CUST_001", "98765": "CUST_002"},
             "Chase Bank": {"Dummy_001": "CUST_003"}
        }
        """

        # Expected Results
        self.expected_trxns = {
            "345723": {
                1: {
                    "Originator_Name": "John",
                    "Originator_Account_ID": "345723",
                    "Originator_Customer_ID": "CUST_001",
                    "Beneficiary_Name": "John",
                    "Beneficiary_Account_ID": "345723",
                    "Beneficiary_Customer_ID": "CUST_001",
                    "Trxn_Channel": "Cash",
                    "Trxn_Date": "2024-01-04",
                    "Trxn_Amount": 5000,
                    "Branch_or_ATM Location": "Main Road,NY"
                },
                2: {
                    "Originator_Name": "John",
                    "Originator_Account_ID": "345723",
                    "Originator_Customer_ID": "CUST_001",
                    "Beneficiary_Name": "Acme,Inc",
                    "Beneficiary_Account_ID": "98765",
                    "Beneficiary_Customer_ID": "CUST_002",
                    "Trxn_Channel": "Wire",
                    "Trxn_Date": "2024-01-06",
                    "Trxn_Amount": 3000,
                    "Branch_or_ATM Location": ""
                },
                3: {
                    "Originator_Name": "John",
                    "Originator_Account_ID": "345723",
                    "Originator_Customer_ID": "CUST_001",
                    "Beneficiary_Name": "Jill",
                    "Beneficiary_Account_ID": "Dummy_001",
                    "Beneficiary_Customer_ID": "CUST_003",
                    "Trxn_Channel": "Check",
                    "Trxn_Date": "2024-01-08",
                    "Trxn_Amount": 1000,
                    "Branch_or_ATM Location": ""
                }
            }
        }

        # Required fields for each transaction
        self.required_keys = {
            "Originator_Name",
            "Originator_Account_ID",
            "Originator_Customer_ID",
            "Beneficiary_Name",
            "Beneficiary_Account_ID",
            "Beneficiary_Customer_ID",
            "Trxn_Channel",
            "Trxn_Date",
            "Trxn_Amount",
            "Branch_or_ATM Location"
        }

        self.summary_prompt = self.agent_config.get("summary_prompt") 
        logging.info("Step 6: Read summary prompt for Transaction Generation Agent")   

        logging.info("Running Transaction Generation Agent with test message...")
        # Generate the final results for all tests to use
        self.results = create_two_agent_chat(
            self.sar_agent,
            self.transaction_generation_agent,
            self.test_message,
            self.summary_prompt
        )

    def test_number_of_accounts_in_results(self):
        """
        Test that the output dictionary includes the expected account ID key.
        In our example, '345723' is the only key in the final dictionary of transactions.
        """
        logging.info("Testing the presence of account ID '345723' in the results...")
        self.assertIn(
            "345723", 
            self.results, 
            "Expected account 345723 to appear in the transactions dictionary."
        )

    def test_number_of_transactions_for_345723(self):
        """
        We expect three transactions in the agent's output for account '345723'
        """
        logging.info("Testing the number of transactions under account 345723...")
        acct_dict = self.results["345723"]
        self.assertEqual(len(acct_dict), 3, "Expected exactly 3 transactions for account 345723.")

    def test_transaction_details(self):
        """
        Verifies that each transaction has all the required fields.
        """
        acct_dict = self.results["345723"]
        for trx_id, trx_data in acct_dict.items():
            logging.info(f"Checking transaction ID = {trx_id}")
            self.assertSetEqual(
                self.required_keys, 
                set(trx_data.keys()),
                f"Transaction {trx_id} does not have the expected set of keys. "
            )

    def test_trxn_attributes(self):
        """
        Validates that the transaction attributes are as expected.
        """
        actual_345723 = self.results["345723"]
        expected_345723 = self.expected_trxns["345723"]

        for txn_id, expected_fields in expected_345723.items():
            with self.subTest(txn_id=txn_id):
                actual_txn = actual_345723[txn_id]
                for field_name, expected_val in expected_fields.items():
                    self.assertEqual(
                        actual_txn[field_name],
                        expected_val,
                        f"Mismatch for field '{field_name}' in transaction {txn_id}."
                    )













