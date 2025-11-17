import pandas as pd
import numpy as np
import torch
import hashlib


# Load the CSV files
input_df = pd.read_csv('net/home/rosca/maxi_model_training/inputs_DlTransmission3.csv')
output_df = pd.read_csv('net/home/rosca/maxi_model_training/outputs_DlTransmission3.csv') # sed -i 's/,$//' outputs_DlTransmission.csv to remove last empty column
# Define identifying and feature columns
slot_cols = ["simulation", "frame", "subframe", "slot"]
flow_cols = ["rnti", "resource_type", "priority", "delay_budget", 
             "available_symbols", "queue_size", "mcs"]

# *************************************************************************
#********************* Remove duplicate data ******************************
# *************************************************************************

# Define the 12 OFDM symbol output columns (sym1–sym12)
out_sym_cols = [f"sym{i}" for i in range(1, 13)]

# Ensure that frame/subframe/slot/simulation identifiers are all strings
# This avoids issues with mixed data types during grouping and hashing.
input_df[slot_cols] = input_df[slot_cols].astype(str)
output_df[slot_cols] = output_df[slot_cols].astype(str)

# Group all input rows (flows) by their unique slot identifiers
# Each group now corresponds to one slot (which can have multiple flows)
inp_groups = input_df.groupby(slot_cols)

# Index the output DataFrame by slot identifiers for fast lookup
out_indexed = output_df.set_index(slot_cols, drop=False)

# Prepare lists to store the unique slot keys and their computed signatures
signatures = []
slot_keys = []

# Iterate through each group of flows (i.e., each slot)
for key, group in inp_groups:
    # Skip this slot if no corresponding output data exists
    if key not in out_indexed.index:
        continue

    # Sort flows within a slot by RNTI to ensure deterministic ordering
    # (different orderings of the same flows should produce the same hash)
    group_sorted = group.sort_values(by=["rnti", "resource_type"], kind="mergesort")

    # Convert each flow’s features to a compact string: "rnti:resource_type:priority:..."
    flow_strs = group_sorted[flow_cols].astype(str).agg(":".join, axis=1).tolist()

    # Combine all flow strings for this slot into a single serialized string
    input_serial = "|".join(flow_strs)

    # Get the corresponding output row (12 symbol allocations)
    out_row = out_indexed.loc[key]

    # Handle the case where multiple rows match the key
    if isinstance(out_row, pd.DataFrame):
        out_row = out_row.iloc[0]

    # Convert the symbol allocations to a comma-separated string: "1,1,2,2,3,3,..."
    output_serial = ",".join(out_row[out_sym_cols].astype(str).tolist())

    # Combine input and output serializations into one long string
    combined = input_serial + "||" + output_serial

    # Compute a SHA-1 hash of the combined string
    # This acts as a compact “signature” that uniquely identifies this slot
    h = hashlib.sha1(combined.encode("utf-8")).hexdigest()

    # Store the slot key (frame, subframe, slot, etc.) and its signature
    signatures.append(h)
    slot_keys.append(key)

# Create a DataFrame mapping each slot to its computed signature
sig_df = pd.DataFrame(slot_keys, columns=slot_cols)
sig_df["signature"] = signatures

# Count how many times each signature appears
dup_counts = sig_df["signature"].value_counts()

# Identify signatures that appear more than once (i.e., duplicate slots)
duplicated_sigs = dup_counts[dup_counts > 1].index.tolist()

print(f"Total slots: {len(sig_df)}, duplicate signature groups: {len(duplicated_sigs)}")

# Keep only the first occurrence of each unique signature
# This removes duplicated slot entries
first_occ = sig_df.drop_duplicates(subset=["signature"], keep="first")

# Extract the list of unique (frame, subframe, slot, simulation) keys to retain
unique_keys = [tuple(x) for x in first_occ[slot_cols].values]

# Build tuple keys for every row in the input and output DataFrames
inp_keys = input_df[slot_cols].apply(lambda row: tuple(row.values), axis=1)
out_keys = output_df[slot_cols].apply(lambda row: tuple(row.values), axis=1)

# Create boolean masks selecting only the rows corresponding to unique slots
keep_inp_mask = inp_keys.isin(unique_keys)
keep_out_mask = out_keys.isin(unique_keys)

# Filter the original data to keep only non-duplicate slots
inp_clean = input_df[keep_inp_mask].copy()
out_clean = output_df[keep_out_mask].copy()

# Define file paths for the cleaned datasets
clean_input_path = "net/home/rosca/maxi_model_training/inputs_DlTransmission_no_duplicates3.csv"
clean_output_path = "net/home/rosca/maxi_model_training/outputs_DlTransmission_no_duplicates3.csv"

# Save the cleaned DataFrames to new CSV files
inp_clean.to_csv(clean_input_path, index=False)
out_clean.to_csv(clean_output_path, index=False)

# Print summary of how many duplicate slots were removed and where data was saved
print(f"Removed {len(sig_df) - len(first_occ)} duplicate slots.")
print(f"Saving cleaned inputs to {clean_input_path} (rows: {len(inp_clean)})")
print(f"Saving cleaned outputs to {clean_output_path} (rows: {len(out_clean)})")

# *************************************************************************
# ********************* End remove duplicate data **************************
# *************************************************************************


# # *************************************************************************
# # ******************** Zero-padding ***************************************
# # *************************************************************************

# # Maximum number of flows per slot
# MAX_FLOWS = 5
# inp_clean = pd.read_csv('net/home/rosca/maxi_model_training/inputs_DlTransmission_no_duplicates3.csv')
# # Output container
# padded_rows = []

# for key, group in inp_clean.groupby(slot_cols):
#     # Sort deterministically by rnti and resource_type
#     group_sorted = group.sort_values(by=["rnti", "resource_type"], kind="mergesort")

#     # Count number of flows in this slot
#     num_flows = len(group_sorted)

#     # If fewer than MAX_FLOWS, append zero-padded rows
#     if num_flows < MAX_FLOWS:
#         # Build padding rows with zeros (same columns as group_sorted)
#         pad_rows = pd.concat(
#             [pd.DataFrame([[*key, 0, 0, 0, 0, 0, 0, 0]], columns=slot_cols + flow_cols)] * (MAX_FLOWS - num_flows),
#             ignore_index=True
#         )
#         group_sorted = pd.concat([group_sorted, pad_rows], ignore_index=True)

#     # Append processed group
#     padded_rows.append(group_sorted)

# # Combine all processed groups into a single DataFrame
# padded_input_df = pd.concat(padded_rows, ignore_index=True)
# for col in slot_cols:
#     padded_input_df[col] = padded_input_df[col].astype(int)
# padded_input_df = padded_input_df.sort_values(slot_cols).reset_index(drop=True)
# padded_input = "net/home/rosca/maxi_model_training/inputs_DlTransmission_padded3.csv"
# padded_input_df.to_csv(padded_input, index=False)

# # ************************************************************************
# # ******************** End zero-padding **********************************
# # ************************************************************************


# # ***********************************************************************
# # ********* Apply z-score normalization to the relevant columns**********
# # ***********************************************************************
# padded_input_df = pd.read_csv('net/home/rosca/maxi_model_training/inputs_DlTransmission_padded3.csv')
# cols_to_zscore = ["priority", "delay_budget", "available_symbols", "queue_size", "mcs"]

# # Mask: exclude padded rows (rnti == 0)
# mask = padded_input_df["rnti"] != 0

# # Conversion to float for normalization
# padded_input_df[cols_to_zscore] = padded_input_df[cols_to_zscore].astype(float)
# mean = padded_input_df.loc[mask, cols_to_zscore].mean()  # Compute the mean
# std = padded_input_df.loc[mask, cols_to_zscore].std()  # Compute the standard deviation
# padded_input_df.loc[mask, cols_to_zscore] = (padded_input_df.loc[mask, cols_to_zscore] - mean) / std  # Apply z-score normalization

# # Save the normalization parameters
# torch.save({'mean': mean, 'std': std}, 'normalization_params_zscore2.pt')

# # ************************************************************************
# # ********* End z-score normalization to the relevant columns*************
# # ************************************************************************

# # Save the processed DataFrame to a new CSV file
# padded_input_df.to_csv('net/home/rosca/maxi_model_training/inputs_DlTransmission_zscore3.csv', index=False)
