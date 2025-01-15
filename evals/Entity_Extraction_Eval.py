import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any
from main import run_agentic_workflow
from utils import read_data

# ============================
# Define SARs and Gold Standards
# ============================

# Example SARs and their gold standard outputs
# In practice, you would load this data from a file or database

# Define a data structure for each SAR
class SAR:
    def __init__(self, sar_name: str, sar_narrative: str, gold_entities: Dict[str, List[str]],
                 gold_account_ids: List[str],
                 gold_acct_to_fi: Dict[str, str],
                 gold_acct_to_cust: Dict[str, str]):
        self.sar_name = sar_name
        self.sar_narrative = sar_narrative,
        self.gold_entities = gold_entities
        self.gold_account_ids = gold_account_ids
        self.gold_acct_to_fi = gold_acct_to_fi
        self.gold_acct_to_cust = gold_acct_to_cust

config_file = 'configs/agents_config.yaml' 
sar_narratives = read_data(train = True)

# Example SARs (Add more SAR instances as needed)
sars = [
    SAR(
        sar_name="sar_train1",
        sar_narrative = sar_narratives[0],
        gold_entities={
            "Individuals": ["John Doe"],
            "Organizations": ["Acme, Inc.", "Kulkutta Building Supply Company"],
            "Financial Institutions": ["Bank of Anan"]
        },
        gold_account_ids=["123456789", "234567891", "3489728"],
        gold_acct_to_fi={
            "123456789": "Dummy_Bank_1",
            "234567891": "Dummy_Bank_1",
            "3489728": "Bank of Anan"
        },
        gold_acct_to_cust={
            "123456789": "John Doe",
            "234567891": "Acme, Inc.",
            "3489728": "Kulkutta Building Supply Company"
        }
    ),
    # Add second SAR
    SAR(
        sar_name="sar_train2",
        sar_narrative = sar_narratives[1],
        gold_entities={
           "Individuals": ["John Doe", "Jane Doe"],
            "Organizations": ["Doe’s Auto Sales"],
            "Financial Institutions": ["XYZ Bank"]
        },
        gold_account_ids= ["1234567", "Dummy_Acct_1"],
        gold_acct_to_fi={
                    "1234567": "Dummy_Bank_1" ,
                    "Dummy_Acct_1": "XYZ Bank"
                },
        gold_acct_to_cust= {
                "1234567": "Doe’s Auto Sales",
                "Dummy_Acct_1": "Doe’s Auto Sales"
                       }
    ),
    # Add third SAR
    SAR(
        sar_name="sar_train3",
        sar_narrative = sar_narratives[2],
        gold_entities={
                    "Individuals": ["John Doe", "Jennifer Doe"],
                    "Organizations": ["Quickie Car Wash"],
                    "Financial Institutions": ["Aussie Bank"]
                },
        gold_account_ids=  ["12345678910", "981012345"],
        gold_acct_to_fi={
                    "12345678910": "Dummy_Bank_1" ,
                    "981012345": "Aussie Bank"
                },
            gold_acct_to_cust=  {
                        "12345678910": "John Doe",
                        "981012345": "Jennifer Doe"
                    }
    ),
    # Add fourth SAR
    SAR(
        sar_name="sar_train4",
        sar_narrative = sar_narratives[3],
        gold_entities={
                        "Individuals": ["Paul Lafonte"],
                        "Organizations": ["Sky Corporation", "Sea Corporation", "Tolinka Inc."],
                        "Financial Institutions": ["Bank of Mainland", "Bank XYZ", "Bank of Poland", "Artsy Bank"]
                    },
        gold_account_ids=   ["54321098", "12345678", "689472", "456781234", "Dummy_Acct_1","Dummy_Acct_2"],
        gold_acct_to_fi={
                    "54321098": "Bank of Mainland",
                    "12345678": "Bank of Mainland",
                    "689472": "Bank XYZ",
                    "456781234": "Artsy Bank",
                    "Dummy_Acct_1": "Bank of Poland",
                     "Dummy_Acct_2": "Bank of Mainland"
                },
        gold_acct_to_cust=  {
                        "54321098": "Sky Corporation",
                        "12345678": "Sea Corporation",
                        "689472": "Tolinka Inc.",
                        "456781234": "Paul Lafonte",
                        "Dummy_Acct_1": "Bank XYZ",
                        "Dummy_Acct_2": "Bank of Poland"
                    }
    ),



]


# ============================
# Evaluation Functions
# ============================

def evaluate_entities(pred_entities: Dict[str, List[str]],
                     gold_entities: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Evaluates the Entities extraction.
    Returns precision, recall, and F1-score for each entity type.
    """
    results = {}
    for entity_type in gold_entities.keys():
        gold_set = set(gold_entities.get(entity_type, []))
        pred_set = set(pred_entities.get(entity_type, []))
        true_positives = len(gold_set & pred_set)
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(gold_set) if gold_set else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
        results[entity_type] = {"precision": precision, "recall": recall, "f1": f1}
    return results

def evaluate_acct_ids(pred_acct_ids: List[str], gold_acct_ids: List[str]) -> Dict[str, float]:
    """
    Evaluates list-based extractions like Account_IDs.
    Returns precision, recall, and F1-score.
    """
    gold_set = set(pred_acct_ids)
    pred_set = set(gold_acct_ids)
    true_positives = len(gold_set & pred_set)
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(gold_set) if gold_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_mappings(pred_mapping: Dict[str, str], gold_mapping: Dict[str, str]) -> Dict[str, float]:
    """
    Evaluates mapping dictionaries like Acct_to_FI and Acct_to_Cust.
    Returns precision, recall, and F1-score based on correct key-value pairs.
    """
    gold_pairs = set(gold_mapping.items())
    pred_pairs = set(pred_mapping.items())
    true_positives = len(gold_pairs & pred_pairs)
    precision = true_positives / len(pred_pairs) if pred_pairs else 0
    recall = true_positives / len(gold_pairs) if gold_pairs else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return {"precision": precision, "recall": recall, "f1": f1}

# ============================
# Main Evaluation Loop
# ============================

def evaluate_sars(sars: List[SAR]) -> pd.DataFrame:
    """
    Evaluates a list of SARs and returns a DataFrame with evaluation metrics.
    """
    # Initialize lists to collect metrics
    entity_metrics = defaultdict(list)
    account_id_metrics = []
    acct_to_fi_metrics = []
    acct_to_cust_metrics = []

    for idx, sar in enumerate(sars):
        print(f"Evaluating SAR {idx+1}/{len(sars)}...")
        # Run the agent workflow
        #pred_output = run_agentic_workflow(sar.sar_narrative, config_file)
        pred_output = {
                    "Entities": {"Individuals": ["John Doe"],
                                 "Organizations": ["Acme, Inc.", "Kulkutta Building Supply Company"],
                                 "Financial Institutions": ["Bank of Anan"]},
                     "Account_IDs":["123456789", "234567891", "3489728"],
                     "Acct_to_FI" : {
                                        "123456789": "Dummy_Bank_1",
                                        "234567891": "Dummy_Bank_1",
                                        "3489728": "Bank of Anan"
                                    },
                     "Acct_to_Cust": {
                                "123456789": "John Doe",
                                "234567891": "Acme, Inc.",
                                "3489728": "Kulkutta Building Supply Company"
                            }
                    
                    }

        # Evaluate Entities
        ent_metrics = evaluate_entities(pred_output.get("Entities", {}),
                                        sar.gold_entities)
        for ent_type, metrics in ent_metrics.items():
            entity_metrics[ent_type].append(metrics)

        # Evaluate Account_IDs
        acct_metrics = evaluate_acct_ids(pred_output.get("Account_IDs", []),
                                      sar.gold_account_ids)
        account_id_metrics.append(acct_metrics)

        # Evaluate Acct_to_FI
        acct_fi_metrics = evaluate_mappings(pred_output.get("Acct_to_FI", {}),
                                           sar.gold_acct_to_fi)
        acct_to_fi_metrics.append(acct_fi_metrics)

        # Evaluate Acct_to_Cust
        acct_cust_metrics = evaluate_mappings(pred_output.get("Acct_to_Cust", {}),
                                             sar.gold_acct_to_cust)
        acct_to_cust_metrics.append(acct_cust_metrics)

    # Aggregate Entity Metrics
    entity_summary = {}
    for ent_type, metrics_list in entity_metrics.items():
        avg_precision = sum(m['precision'] for m in metrics_list) / len(metrics_list)
        avg_recall = sum(m['recall'] for m in metrics_list) / len(metrics_list)
        avg_f1 = sum(m['f1'] for m in metrics_list) / len(metrics_list)
        entity_summary[ent_type] = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }

    # Aggregate Account_IDs Metrics
    avg_acct_precision = sum(m['precision'] for m in account_id_metrics) / len(account_id_metrics)
    avg_acct_recall = sum(m['recall'] for m in account_id_metrics) / len(account_id_metrics)
    avg_acct_f1 = sum(m['f1'] for m in account_id_metrics) / len(account_id_metrics)
    account_id_summary = {
        "Account_IDs": {
            "precision": avg_acct_precision,
            "recall": avg_acct_recall,
            "f1": avg_acct_f1
        }
    }

    # Aggregate Acct_to_FI Metrics
    avg_fi_acct_precision = sum(m['precision'] for m in acct_to_fi_metrics) / len(acct_to_fi_metrics)
    avg_fi_acct_recall = sum(m['recall'] for m in acct_to_fi_metrics) / len(acct_to_fi_metrics)
    avg_fi_acct_f1 = sum(m['f1'] for m in acct_to_fi_metrics) / len(acct_to_fi_metrics)
    fi_to_acct_summary = {
        "Acct_to_FI": {
            "precision": avg_fi_acct_precision,
            "recall": avg_fi_acct_recall,
            "f1": avg_fi_acct_f1
        }
    }

    # Aggregate Acct_to_Cust Metrics
    avg_acct_cust_precision = sum(m['precision'] for m in acct_to_cust_metrics) / len(acct_to_cust_metrics)
    avg_acct_cust_recall = sum(m['recall'] for m in acct_to_cust_metrics) / len(acct_to_cust_metrics)
    avg_acct_cust_f1 = sum(m['f1'] for m in acct_to_cust_metrics) / len(acct_to_cust_metrics)
    acct_to_cust_summary = {
        "Acct_to_Cust": {
            "precision": avg_acct_cust_precision,
            "recall": avg_acct_cust_recall,
            "f1": avg_acct_cust_f1
        }
    }

    # Combine all summaries
    combined_summary = {**entity_summary, **account_id_summary,
                        **fi_to_acct_summary, **acct_to_cust_summary}

    # Convert to DataFrame for better visualization
    metrics_df = pd.DataFrame(combined_summary).T

    return metrics_df

# ============================
# Run Evaluation
# ============================

if __name__ == "__main__":
    # Evaluate the SARs
    results_df = evaluate_sars(sars)

    # Display the results
    print("\nEvaluation Metrics:")
    print(results_df)

    # Optionally, save the results to a CSV file
    results_df.to_csv("evaluation_metrics.csv")
    print("\nEvaluation metrics saved to 'evaluation_metrics.csv'")
