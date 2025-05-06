
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from utils import flatten_nested_mapping,approximate_match_ratio,concatenate_trxn_sets
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def evaluate_dict_keys(narratives: Dict[str, str], gold_narratives: Dict[str, str]) -> Dict[str, float]:
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

def count_transaction_sets(transactions_dict):
    """
    Counts the total number of transaction sets in the provided dictionary.
    Only counts keys matching the pattern 'Trxn_Set_'.
    """
    total_count = 0
    for value in transactions_dict.values():
        if isinstance(value, dict):
            # Count all sub-keys
            total_count += len(value.keys())
        else:
            # It's a single narrative string for that account
            total_count += 1
    return total_count

def evaluate_transaction_sets(pred_dict,gold_dict):
    """
    Evaluates whether the right number of trxn sets have been extracted
    """
    observed = count_transaction_sets(pred_dict)
    expected = count_transaction_sets(gold_dict)

    return {
        "observed": observed,
        "expected":expected
    }
# ============================
# Main Evaluation Loop
# ============================



def compare_sar_details(
    ground_truth_sars: List,
    predicted_sars: List
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates a list of ground truth SARs and predicted SARs (already generated externally),
    returning two DataFrames:

    1) A metrics DataFrame that aggregates precision, recall, F1, etc.
    2) A narrative match ratio DataFrame.

    :param ground_truth_sars: A list of ground-truth SAR objects, each containing
                              gold-standard fields, such as:
                              gold_entities, gold_account_ids, gold_acct_to_fi,
                              gold_acct_to_cust, gold_fi_to_acct_to_cust, gold_narrative, etc.
    :param predicted_sars: A list of predicted SAR dictionaries (or objects), each containing
                           predicted fields, such as:
                           "Entities", "Account_IDs", "Acct_to_FI", "Acct_to_Cust",
                           "FI_to_Acct_to_Cust", "Narrative", etc.
    :return: (metrics_df, narratives_df)
             metrics_df: DataFrame with one row per SAR and a final "Average" row,
                         showing various precision/recall/f1 metrics, plus transaction set counts.
             narratives_df: DataFrame with account-level narrative match ratios.
    """

    metric_rows = []
    narrative_rows = []

    for idx, (gt_sar, pred_sar) in enumerate(zip(ground_truth_sars, predicted_sars)):
        logging.info(f"Evaluating SAR {idx+1}/{len(ground_truth_sars)}...")

        # Instead of generating predictions here (e.g., run_agentic_workflow),
        # we assume pred_sar already contains all relevant predicted data.
        # For example:
        # pred_sar = {
        #   "Entities": {...},
        #   "Account_IDs": [...],
        #   "Acct_to_FI": {...},
        #   "Acct_to_Cust": {...},
        #   "FI_to_Acct_to_Cust": {...},
        #   "Narrative": {...}
        # }

        # ===== Evaluate Entities =====
        ent_metrics = evaluate_entities(
            pred_sar.get("Entities", {}),
            gt_sar.gold_entities
        )

        # ===== Evaluate Account_IDs =====
        acct_metrics = evaluate_acct_ids(
            pred_sar.get("Account_IDs", []),
            gt_sar.gold_account_ids
        )

        # ===== Evaluate Acct_to_FI =====
        acct_fi_metrics = evaluate_mappings(
            pred_sar.get("Acct_to_FI", {}),
            gt_sar.gold_acct_to_fi
        )

        # ===== Evaluate Acct_to_Cust =====
        acct_cust_metrics = evaluate_mappings(
            pred_sar.get("Acct_to_Cust", {}),
            gt_sar.gold_acct_to_cust
        )

        # ===== Evaluate FI_to_Acct_to_Cust =====
        fi_acct_cust_metrics = evaluate_nested_mapping(
            pred_sar.get("FI_to_Acct_to_Cust", {}),
            gt_sar.gold_fi_to_acct_to_cust
        )

        # ===== Check that narratives are extracted for expected accounts =====
        accts_w_narrative_metrics = evaluate_dict_keys(
            pred_sar.get("Narrative", {}),
            gt_sar.gold_narrative
        )

        # ===== Evaluate expected number of transaction sets =====
        trxn_set_metrics = evaluate_transaction_sets(
            pred_sar.get("Narrative", {}),
            gt_sar.gold_narrative
        )

        # ----- Build one row of results for this SAR -----
        row_data = {
            "SAR_index": idx + 1  # Human-friendly numbering
        }

        # Entities metrics: possibly multiple entity types
        for entity_type, metrics_dict in ent_metrics.items():
            safe_type = entity_type.replace(" ", "_")
            row_data[f"{safe_type}_precision"] = metrics_dict["precision"]
            row_data[f"{safe_type}_recall"] = metrics_dict["recall"]
            row_data[f"{safe_type}_f1"] = metrics_dict["f1"]

        # Account_IDs metrics
        row_data["Account_IDs_precision"] = acct_metrics["precision"]
        row_data["Account_IDs_recall"]   = acct_metrics["recall"]
        row_data["Account_IDs_f1"]       = acct_metrics["f1"]

        # FI_to_Acct metrics
        row_data["FI_to_Acct_precision"] = acct_fi_metrics["precision"]
        row_data["FI_to_Acct_recall"]   = acct_fi_metrics["recall"]
        row_data["FI_to_Acct_f1"]       = acct_fi_metrics["f1"]

        # Acct_to_Cust metrics
        row_data["Acct_to_Cust_precision"] = acct_cust_metrics["precision"]
        row_data["Acct_to_Cust_recall"]   = acct_cust_metrics["recall"]
        row_data["Acct_to_Cust_f1"]       = acct_cust_metrics["f1"]

        # FI_Acct_Cust_ID metrics
        row_data["FI_Acct_Cust_ID_precision"] = fi_acct_cust_metrics["precision"]
        row_data["FI_Acct_Cust_ID_recall"]   = fi_acct_cust_metrics["recall"]
        row_data["FI_Acct_Cust_ID_f1"]       = fi_acct_cust_metrics["f1"]

        # Narrative Key Metrics
        row_data["accts_in_narrative_precision"] = accts_w_narrative_metrics["precision"]
        row_data["accts_in_narrative_recall"]   = accts_w_narrative_metrics["recall"]
        row_data["accts_in_narrative_f1"]       = accts_w_narrative_metrics["f1"]

        row_data["N_observed_trxn_sets"] = trxn_set_metrics["observed"]
        row_data["N_expected_trxn_sets"] = trxn_set_metrics["expected"]

        metric_rows.append(row_data)

        # ----- Evaluate narrative similarity -----
        gold_narr = concatenate_trxn_sets(gt_sar.gold_narrative)
        pred_narr = concatenate_trxn_sets(pred_sar.get("Narrative", {}))
        all_accts = set(gold_narr.keys()) | set(pred_narr.keys())

        for acct_id in all_accts:
            g_text = gold_narr.get(acct_id, "")
            p_text = pred_narr.get(acct_id, "")
            ratio = approximate_match_ratio(g_text, p_text)
            narrative_rows.append({
                "SAR_index": (idx + 1),
                "Account_ID": acct_id,
                "narrative_match_ratio": ratio
            })

    # ----- Convert collected metrics into DataFrames -----
    metrics_df = pd.DataFrame(metric_rows)

    # Optionally compute the average row if we have numeric columns
    if not metrics_df.empty:
        avg_row = metrics_df.mean(numeric_only=True).to_dict()
        avg_row["SAR_index"] = "Average"
        avg_row_df = pd.DataFrame([avg_row])
        metrics_df = pd.concat([metrics_df, avg_row_df], ignore_index=True)

    narratives_df = pd.DataFrame(narrative_rows)
    return metrics_df, narratives_df






def compare_trxns(df: pd.DataFrame, expected_trxns: Dict[str, Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """
    Compare actual transactions in `df` with expected transaction sets in `expected_trxns`.
    For each transaction set:
      - Compute the percentage mismatches (Amount_pct_diff, Count_pct_diff).
      - Identify missing/extra channels, missing/extra locations, etc.
      - Compute the fraction of transactions whose dates and amounts fall within expected ranges.

    Returns
    -------
    pd.DataFrame
        A DataFrame with comparison metrics for each (SAR ID, Trxn_Set).
    """

    # Ensure Trxn_Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["Trxn_Date"]):
        df["Trxn_Date"] = pd.to_datetime(df["Trxn_Date"])

    results = []

    for sar_id, trx_sets in expected_trxns.items():
        for set_id, expected in trx_sets.items():
            # Filter by Originator and Beneficiary
            sub_df = df[
                (df["Originator_Account_ID"] == expected["Originator_Account_ID"]) &
                (df["Beneficiary_Account_ID"] == expected["Beneficiary_Account_ID"])
            ]

            if sub_df.empty:
                # Nothing matches => fill defaults
                results.append({
                    "SAR ID": sar_id,
                    "Trxn_Set_ID": set_id,
                    "Amount_pct_diff": "",
                    "Count_pct_diff": "",
                    "N_trxns_in_date_range": "",
                    "Missing_channels": "",
                    "Extra_channels": "",
                    "Channels_match": False,
                    "Missing_locations": "",
                    "Extra_locations": "",
                    "Locations_match": False,
                    "Perc_ind_amt_in_range": 0
                })
                continue

            # -----------------------------------------------------
            # 1. Amount & Count mismatch
            # -----------------------------------------------------
            expected_amount = float(expected["Total_Amount"])
            expected_count = int(expected["N_trxns"])

            # If 'Trxn_Amount' is entirely empty (all NaN), treat as 0
            if sub_df["Trxn_Amount"].dropna().empty:
                actual_amount = 0
                actual_count = 0
            else:
                actual_amount = sub_df["Trxn_Amount"].sum()
                actual_count = len(sub_df)

            # Amount % diff
            if expected_amount == 0:
                amount_pct_diff = 0.0 if actual_amount == 0 else 100.0
            else:
                amount_pct_diff = ((actual_amount - expected_amount) / expected_amount) * 100.0

            # Count % diff
            if expected_count == 0:
                count_pct_diff = 0.0 if actual_count == 0 else 100.0
            else:
                count_pct_diff = ((actual_count - expected_count) / expected_count) * 100.0

            # -----------------------------------------------------
            # 2. Date Range fraction
            # -----------------------------------------------------
            if sub_df["Trxn_Date"].dropna().empty:
                # If date column is empty => can't determine date coverage
                n_trxns_in_date_range = 0
            else:
                min_date = pd.to_datetime(expected["Min_Date"])
                max_date = pd.to_datetime(expected["Max_Date"])
                if actual_count == 0:
                    n_trxns_in_date_range = 0
                else:
                    in_range_mask = (sub_df["Trxn_Date"] >= min_date) & (sub_df["Trxn_Date"] <= max_date)
                    n_trxns_in_date_range = in_range_mask.sum() / actual_count

            # -----------------------------------------------------
            # 3. Channel checks
            # -----------------------------------------------------
            expected_channels = set(expected["Trxn_Type"])
            if sub_df["Trxn_Channel"].dropna().empty:
                # If every row is empty for channel => no actual channels
                actual_channels = set()
            else:
                actual_channels = set(sub_df["Trxn_Channel"].dropna().unique())

            missing_channels = list(expected_channels - actual_channels)
            extra_channels = list(actual_channels - expected_channels)
            channels_match = (actual_channels == expected_channels)

            # -----------------------------------------------------
            # 4. Location checks
            # -----------------------------------------------------
            expected_locations = set(expected.get("Branch_or_ATM_Location", []))
            if sub_df["Branch_or_ATM_Location"].dropna().empty:
                actual_locations = set()
            else:
                actual_locations = set(sub_df["Branch_or_ATM_Location"].dropna().unique())

            missing_locations = list(expected_locations - actual_locations)
            extra_locations = list(actual_locations - expected_locations)
            locations_match = (actual_locations == expected_locations)

            # -----------------------------------------------------
            # 5. Individual amount range
            # -----------------------------------------------------
            if sub_df["Trxn_Amount"].dropna().empty or actual_count == 0:
                perc_ind_amt_in_range = 0
            else:
                min_ind_amt = expected["Min_Ind_Amt"]
                max_ind_amt = expected["Max_Ind_Amt"]
                in_amt_range_mask = (sub_df["Trxn_Amount"] >= min_ind_amt) & (sub_df["Trxn_Amount"] <= max_ind_amt)
                perc_ind_amt_in_range = in_amt_range_mask.sum() / actual_count

            # -----------------------------------------------------
            # 6. Add results for this row
            # -----------------------------------------------------
            results.append({
                "SAR ID": sar_id,
                "Trxn_Set_ID": set_id,
                "Amount_pct_diff": amount_pct_diff,
                "Count_pct_diff": count_pct_diff,
                "N_trxns_in_date_range": n_trxns_in_date_range,
                "Missing_channels": missing_channels,
                "Extra_channels": extra_channels,
                "Channels_match": channels_match,
                "Missing_locations": missing_locations,
                "Extra_locations": extra_locations,
                "Locations_match": locations_match,
                "Perc_ind_amt_in_range": perc_ind_amt_in_range
            })

    # Build final DataFrame
    results_df = pd.DataFrame(results)

    # Optional: reorder columns
    col_order = [
        "SAR ID", "Trxn_Set_ID", "Amount_pct_diff", "Count_pct_diff",
        "N_trxns_in_date_range", "Missing_channels", "Extra_channels",
        "Channels_match", "Missing_locations", "Extra_locations",
        "Locations_match", "Perc_ind_amt_in_range"
    ]
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order]

    return results_df

