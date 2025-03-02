
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from utils import flatten_nested_mapping ,approximate_match_ratio

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
    gold_set = set(gold_acct_ids)
    pred_set = set(pred_acct_ids)
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


def evaluate_dict_keys(narratives: Dict[str, float], gold_narratives: Dict[str, float]) -> Dict[str, float]:
    """
    Evaluates whether the keys of two dictionaries match
    Returns precision, recall, and F1-score.
    """
    gold_set = set(gold_narratives.keys())
    pred_set = set(narratives.keys())
    true_positives = len(gold_set & pred_set)
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(gold_set) if gold_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_nested_mapping(pred_dict, gold_dict):
    """
    Evaluates how accurately the predicted nested mapping matches the gold standard.
    Returns a dictionary with precision, recall, and F1-score.
    """
    # Flatten each nested dictionary into sets of (bank, account, customer_id)
    pred_set = flatten_nested_mapping(pred_dict)
    gold_set = flatten_nested_mapping(gold_dict)
    
    # Compute True Positives, etc.
    true_positives = len(pred_set & gold_set)
    pred_count = len(pred_set)
    gold_count = len(gold_set)
    
    precision = true_positives / pred_count if pred_count > 0 else 0.0
    recall = true_positives / gold_count if gold_count > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ============================
# Main Evaluation Loop
# ============================

def evaluate_sars(sars: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates a list of SARs and returns a DataFrame with evaluation metrics.
    """

     # We'll collect the results in a list of dicts; 
    # each dict will become one row in the final DataFrame.
    metric_rows = []

     # This will hold rows for the narrative comparison
    narrative_rows = []

    for idx, sar in enumerate(sars):
        print(f"Evaluating SAR {idx+1}/{len(sars)}...")
        # Run the agent workflow
        #pred_output = run_agentic_workflow(sar.sar_narrative, config_file)
        pred_output = {
                    "Entities": {
                                "Individuals": ["John Doe"],
                                "Organizations": ["Acme, Inc.", "Kulkutta Building Supply Company"],
                                "Financial Institutions": ["Bank of Anan"]
                            },
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
                            },
                     'FI_to_Acct_to_Cust': {'Dummy_Bank_1': {'12345-6789': 'CUST_001','23456-7891': 'CUST_002'},
                                            'Bank of Anan': {'3489728': 'CUST_003'}},
                      'Narrative': {'12345-6789': ' Between January 2 and March 17, 2003, 13 deposits totaling approximately $50,000 were posted to the account, consisting of cash, checks, and money orders, with amounts ranging from $1,500 to $9,500. Third-party out of state checks and money orders were also deposited. Between January 17, 2003, and March 21, 2003, John Doe originated nine wires totaling $225,000 to the Bank of Anan in Dubai, UAE, to benefit Kulkutta Building Supply Company, account #3489728.',
                                    '23456-7891': ' Between January 2 and March 17, 2003, 33 deposits totaling approximately $275,000 were made to the account, consisting of cash, checks, and money orders. Individual amounts ranged between $4,446 and $9,729; 22 of 33 deposits were between $9,150 and $9,980. In nine instances where cash deposits were made to both accounts on the same day, combined deposits exceeded $10,000. Currency transaction reports were filed with the IRS for daily transactions exceeding $10,000. The bank identified Acme, Inc. as providing remittance services to the Middle East, including Iran, without being a licensed money wire transfer business.',
                                     '3489728': "Nine wire transfers totaling $225,000 were sent from John Doe's personal account #12345-6789 at Dummy_Bank_1 to Kulkutta Building Supply Company, account #3489728 at the Bank of Anan in Dubai, UAE, between January 17, 2003, and March 21, 2003."}

                    }

        # Evaluate Entities
        ent_metrics = evaluate_entities(pred_output.get("Entities", {}),
                                        sar.gold_entities)

        # Evaluate Account_IDs
        acct_metrics = evaluate_acct_ids(pred_output.get("Account_IDs", []),
                                      sar.gold_account_ids)
    

        # Evaluate Acct_to_FI
        acct_fi_metrics = evaluate_mappings(pred_output.get("Acct_to_FI", {}),
                                           sar.gold_acct_to_fi)


        # Evaluate Acct_to_Cust
        acct_cust_metrics = evaluate_mappings(pred_output.get("Acct_to_Cust", {}),
                                             sar.gold_acct_to_cust)


        # Evaluate Fi to Acct to CUST ID
        print(pred_output.get("FI_to_Acct_to_Cust",{}))
        print("\n")
        print(sar.gold_fi_to_acct_to_cust)
        fi_acct_cust_metrics = evaluate_nested_mapping(pred_output.get("FI_to_Acct_to_Cust",{}), sar.gold_fi_to_acct_to_cust)

        #Evaluate if narratives have been extracted for expected accounts
        accts_w_narrative_metrics = evaluate_dict_keys(pred_output.get("Narrative",{}), sar.gold_narrative)

        
         # ========== Build One Row of Results for This SAR ==========
        row_data = {
            "SAR_index": idx + 1  # Human-friendly numbering
        }

        # Entities metrics: we may have multiple types, e.g. "Individuals", "Organizations", etc.
        for entity_type, metrics_dict in ent_metrics.items():
            # Replace spaces in entity_type to avoid awkward column names
            safe_type = entity_type.replace(" ", "_")
            row_data[f"{safe_type}_precision"] = metrics_dict["precision"]
            row_data[f"{safe_type}_recall"] = metrics_dict["recall"]
            row_data[f"{safe_type}_f1"] = metrics_dict["f1"]

        # Account_IDs metrics
        row_data["Account_IDs_precision"] = acct_metrics["precision"]
        row_data["Account_IDs_recall"] = acct_metrics["recall"]
        row_data["Account_IDs_f1"] = acct_metrics["f1"]

        # FI_to_Acct metrics
        row_data["FI_to_Acct_precision"] = acct_fi_metrics["precision"]
        row_data["FI_to_Acct_recall"] = acct_fi_metrics["recall"]
        row_data["FI_to_Acct_f1"] = acct_fi_metrics["f1"]

        # Acct_to_Cust metrics
        row_data["Acct_to_Cust_precision"] = acct_cust_metrics["precision"]
        row_data["Acct_to_Cust_recall"] = acct_cust_metrics["recall"]
        row_data["Acct_to_Cust_f1"] = acct_cust_metrics["f1"]

        #Acct_to_Cust_ID metrics
        row_data["FI_Acct_Cust_ID_precision"] = fi_acct_cust_metrics["precision"]
        row_data["FI_Acct_Cust_ID_recall"] = fi_acct_cust_metrics["recall"]
        row_data["FI_Acct_Cust_ID_f1"] = fi_acct_cust_metrics["f1"]

        #Narrative Key Metrics
        row_data["accts_in_narrative_precision"] = accts_w_narrative_metrics["precision"]
        row_data["accts_in_narrative_recall"] = accts_w_narrative_metrics["recall"]
        row_data["accts_in_narrative_f1"] = accts_w_narrative_metrics["f1"]

        # Append the row to our list
        metric_rows.append(row_data)

     # 6) Evaluate Narrative Similarity
        gold_narr = sar.gold_narrative  # {acct -> gold text}
        pred_narr = pred_output.get("Narrative", {})  # {acct -> predicted text}
        all_accts = set(gold_narr.keys()) | set(pred_narr.keys())

        for acct_id in all_accts:
            g_text = gold_narr.get(acct_id, "")
            p_text = pred_narr.get(acct_id, "")
            ratio = approximate_match_ratio(g_text, p_text)
            narrative_rows.append({
                "SAR_index": idx,
                "Account_ID": acct_id,
                "narrative_match_ratio": ratio
            })


    # 6. Convert the per-SAR results into a DataFrame
    metrics_df = pd.DataFrame(metric_rows)

    # 7. Add a final row with the average of all numeric columns
    if not metrics_df.empty:
        avg_row = metrics_df.mean(numeric_only=True).to_dict()
        avg_row["SAR_index"] = "Average"
        
        # Create a new one-row DataFrame and concatenate
        avg_row_df = pd.DataFrame([avg_row])
        metrics_df = pd.concat([metrics_df, avg_row_df], ignore_index=True)

    # 8) Convert the narrative match results to a separate DataFrame
    narratives_df = pd.DataFrame(narrative_rows)

    return metrics_df, narratives_df

