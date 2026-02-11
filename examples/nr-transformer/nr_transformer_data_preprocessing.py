import pandas as pd
import torch
import hashlib
import numpy as np


# Load the CSV files
input_df = pd.read_csv('~/maxi_model_training/training_dataset6/inputs_DlTransmission-trimmed-noheaderrows.csv')
output_df = pd.read_csv('~/maxi_model_training/training_dataset6/outputs_DlTransmission-trimmed-noheaderrows.csv') # sed -i 's/,$//' outputs_DlTransmission.csv to remove last empty column
# Define identifying and feature columns
slot_cols = ["simulation", "frame", "subframe", "slot"]
flow_cols = ["rnti", "priority", "is_DC_GBR", "PFmetric", "delay_factor"]

# "resource_type","available_symbols", "queue_size", "mcs", "potT", "avgT", "bandwidth"

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
    group_sorted = group.sort_values(by=["priority","rnti"], kind="mergesort")

    # Convert each flow’s features to a compact string: "rnti:resource_type:priority:..."
    flow_strs = group_sorted[flow_cols].astype(str).agg(":".join, axis=1).tolist()

    # Combine all flow strings for this slot into a single serialized string
    input_serial = "|".join(flow_strs)

    # Get the corresponding output row (12 symbol allocations)
    out_row = out_indexed.loc[key]

    # Handle the case where multiple rows match the key
    if isinstance(out_row, pd.DataFrame):
        raise RuntimeError(f"Multiple output rows found for slot key {key}")

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
clean_input_path = "inputs_DlTransmission_no_duplicates_test.csv"
clean_output_path = "outputs_DlTransmission_no_duplicates_test.csv"

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


# *************************************************************************
# ******************** Zero-padding ***************************************
# *************************************************************************

# Maximum number of flows per slot
MAX_FLOWS = 5
inp_clean = pd.read_csv('inputs_DlTransmission_no_duplicates.csv')
# Output container
padded_rows = []

for key, group in inp_clean.groupby(slot_cols):
    # Sort deterministically by resource_type and rnti
    group_sorted = group.sort_values(by=["priority", "rnti"], kind="mergesort")

    # Count number of flows in this slot
    num_flows = len(group_sorted)

    # If fewer than MAX_FLOWS, append zero-padded rows
    if num_flows < MAX_FLOWS:
        assert (group_sorted["rnti"].iloc[num_flows:] == 0).all(), f"Padding error in slot {key}: non-zero rnti in padded rows"
        # Build padding rows with zeros (same columns as group_sorted)
        pad_rows = pd.concat(
            [pd.DataFrame([[*key, 0, 0, 0, 0, 0]], columns=slot_cols + flow_cols)] * (MAX_FLOWS - num_flows),
            ignore_index=True
        )
        group_sorted = pd.concat([group_sorted, pad_rows], ignore_index=True)

    # Append processed group
    padded_rows.append(group_sorted)

# Combine all processed groups into a single DataFrame
padded_input_df = pd.concat(padded_rows, ignore_index=True)
for col in slot_cols:
    padded_input_df[col] = padded_input_df[col].astype(int)
padded_input_df = padded_input_df.sort_values(slot_cols).reset_index(drop=True)
padded_input = "inputs_DlTransmission_padded.csv"
padded_input_df.to_csv(padded_input, index=False)

# ************************************************************************
# ******************** End zero-padding **********************************
# ************************************************************************


# ***********************************************************************
# ********* Apply z-score normalization to the relevant columns**********
# ***********************************************************************
padded_input_df = pd.read_csv('inputs_DlTransmission_padded.csv')
cols_to_zscore = ["priority", "PFmetric", "delay_factor"]

# Mask: exclude padded rows (rnti == 0)
mask = padded_input_df["rnti"] != 0

# Conversion to float for normalization
padded_input_df[cols_to_zscore] = padded_input_df[cols_to_zscore].astype(float)
mean = padded_input_df.loc[mask, cols_to_zscore].mean()  # Compute the mean
std = padded_input_df.loc[mask, cols_to_zscore].std()  # Compute the standard deviation

mean_tensor = torch.tensor([mean[col] for col in cols_to_zscore], dtype=torch.float32)
std_tensor = torch.tensor([std[col] for col in cols_to_zscore], dtype=torch.float32)
padded_input_df.loc[mask, cols_to_zscore] = (padded_input_df.loc[mask, cols_to_zscore] - mean) / std  # Apply z-score normalization

assert not padded_input_df.loc[mask, cols_to_zscore].isna().any().any()
assert not np.isinf(padded_input_df.loc[mask, cols_to_zscore]).any().any()
# Save the normalization parameters
torch.save({'mean': mean_tensor, 'std': std_tensor, "cols": cols_to_zscore}, 'normalization_params.pt')

# ************************************************************************
# ********* End z-score normalization to the relevant columns*************
# ************************************************************************

# Save the processed DataFrame to a new CSV file
padded_input_df.to_csv('inputs_DlTransmission_zscore.csv', index=False)


# # ************** Split data into training, validation, and test sets ****************
# input_df = pd.read_csv('inputs_DlTransmission_zscore.csv')
# output_df = pd.read_csv('outputs_DlTransmission_no_duplicates.csv')

# # Unique simulations
# all_sims_input = sorted(input_df['simulation'].unique())
# all_sims_output = sorted(output_df['simulation'].unique())
# assert len(all_sims_input) == 72
# assert len(all_sims_output) == 72
# assert all_sims_input == all_sims_output

# groups = {
#     "3flows": list(range(0, 18)),
#     "4flows": list(range(18, 36)),
#     "5flows_A": list(range(36, 54)),
#     "5flows_B": list(range(54, 72)),
# }

# rng = np.random.default_rng(seed=42)

# train_sims = []
# val_sims = []
# test_sims = []

# for name, sims in groups.items():
#     sims = np.array(sims)
#     rng.shuffle(sims)

#     n = len(sims)
#     n_train = int(0.6 * n)
#     n_val = int(0.2 * n)

#     train_sims.extend(sims[:n_train])
#     val_sims.extend(sims[n_train:n_train + n_val])
#     test_sims.extend(sims[n_train + n_val:])

# train_input = input_df[input_df['simulation'].isin(train_sims)].copy()
# val_input = input_df[input_df['simulation'].isin(val_sims)].copy()
# test_input = input_df[input_df['simulation'].isin(test_sims)].copy()

# train_output = output_df[output_df['simulation'].isin(train_sims)].copy()
# val_output = output_df[output_df['simulation'].isin(val_sims)].copy()
# test_output = output_df[output_df['simulation'].isin(test_sims)].copy()

# train_input.to_csv('inputs_DlTransmission_train.csv', index=False)
# val_input.to_csv('inputs_DlTransmission_val.csv', index=False)
# test_input.to_csv('inputs_DlTransmission_test.csv', index=False)

# train_output.to_csv('outputs_DlTransmission_train.csv', index=False)
# val_output.to_csv('outputs_DlTransmission_val.csv', index=False)
# test_output.to_csv('outputs_DlTransmission_test.csv', index=False)

# assert set(train_sims).isdisjoint(set(val_sims))
# assert set(train_sims).isdisjoint(set(test_sims))
# assert set(val_sims).isdisjoint(set(test_sims))

# def count_group(sims, start, end):
#     return sum(start <= s < end for s in sims)

# for name, (a, b) in zip(
#     ["3flows", "4flows", "5A", "5B"],
#     [(0,18), (18,36), (36,54), (54,72)]
# ):
#     print(name,
#           count_group(train_sims, a, b),
#           count_group(val_sims, a, b),
#           count_group(test_sims, a, b))
    
# assert set(train_input[slot_cols].apply(tuple, axis=1)) == \
#        set(train_output[slot_cols].apply(tuple, axis=1))
# # -------------------------------------------------------------------------

# df_in = pd.read_csv("/home/maximilianrosca/Downloads/inputs_DlTransmission-trimmed-noheaderrows.csv")
# df_out = pd.read_csv("/home/maximilianrosca/Downloads/outputs_DlTransmission-trimmed-noheaderrows.csv")

# key_cols = ["simulation", "frame", "subframe", "slot"]

# keys_in = set(map(tuple, df_in[key_cols].drop_duplicates().values))
# keys_out = set(map(tuple, df_out[key_cols].drop_duplicates().values))

# only_in_input = keys_in - keys_out
# only_in_output = keys_out - keys_in

# print(f"Keys only in input: {len(only_in_input)}")
# print(f"Keys only in output: {len(only_in_output)}")

# missing_simulations = sorted(k[0] for k in only_in_output)

# print("Simulations present in OUTPUT but missing in INPUT:")
# print(missing_simulations)

# keep_cols = [
#     "simulation",
#     "frame",
#     "subframe",
#     "slot",
#     "rnti",
#     "priority",
#     "is_DC_GBR",
#     "PFmetric",
#     "delay_factor",
# ]

# df_trimmed = df_in[keep_cols]

# # Save to a new file
# df_trimmed.to_csv(
#     "/home/maximilianrosca/Downloads/inputs_DlTransmission-trimmed.csv",
#     index=False
# )

# df = pd.read_csv("/home/maximilianrosca/Downloads/outputs_DlTransmission.csv")

# # Drop first 3 rows
# df_cut = df.iloc[1:].reset_index(drop=True)

# # Save to new file (or overwrite if you want)
# df_cut.to_csv(
#     "outputs_DlTransmission-trimmed-noheaderrows.csv",
#     index=False
# )