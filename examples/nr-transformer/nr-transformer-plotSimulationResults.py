import pandas as pd     
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Model Training/Validation Accuracy Plots ----------------------------------
# --------------------------------------------------------------------------------------------------------------------

df_full_size = pd.read_csv("~/Masterarbeit/Results/Basic_model/training_validation_log_file_basic_full.csv")
df_66_size = pd.read_csv("~/Masterarbeit/Results/Basic_model/training_validation_log_file_basic_66.csv")
df_33_size = pd.read_csv("~/Masterarbeit/Results/Basic_model/training_validation_log_file_basic_33.csv")

# plt.plot(df_full_size['epoch'], df_full_size['train_acc'], color='blue', label='Dataset1: ~1M')
# plt.plot(df_66_size['epoch'], df_66_size['train_acc'], color='orange', label='Dataset2: ~660k')
# plt.plot(df_33_size['epoch'], df_33_size['train_acc'], color='green', label='Dataset3: ~330k')
# plt.plot(df_full_size['epoch'], df_full_size['val_acc'], color='blue', label='Dataset1: ~1M')
# plt.plot(df_66_size['epoch'], df_66_size['val_acc'], color='orange', label='Dataset2: ~660k')
# plt.plot(df_33_size['epoch'], df_33_size['val_acc'], color='green', label='Dataset3: ~330k')
# plt.plot(df_full_size['epoch'], df_full_size['train_acc'], color='blue', label='Train Acc 1M')
# plt.plot(df_full_size['epoch'], df_full_size['val_acc'], color='orange', label='Val Acc 1M')
plt.plot(df_66_size['epoch'], df_66_size['train_acc'], color='green', label='Train Acc 660k')
plt.plot(df_66_size['epoch'], df_66_size['val_acc'], color='red', label='Val Acc 660k')
# plt.plot(df_33_size['epoch'], df_33_size['train_acc'], color='purple', label='Train Acc 330k')
# plt.plot(df_33_size['epoch'], df_33_size['val_acc'], color='brown', label='Val Acc 330k')
plt.yticks(np.arange(80, 105, 5))
plt.xlabel("Epochs")
plt.ylabel("Training/Validation Accuracy [%]")
plt.title("Training/Validation Accuracy over Epochs for Dataset2 (~660k) - Model 1")
plt.grid(True)
plt.legend()
plt.show()


# fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# # ----------------------
# # Subplot 1 — 1M dataset
# # ----------------------
# axes[0].plot(df_full_size['epoch'], df_full_size['train_acc'], color='blue', linewidth=2, alpha=0.7, label='Train Acc 1M')
# axes[0].plot(df_full_size['epoch'], df_full_size['val_acc'],   color='orange', linewidth=2, alpha=0.7, label='Val Acc 1M')
# axes[0].set_ylabel("Accuracy [%]")
# axes[0].set_title("1M Dataset - Model 2")
# axes[0].set_yticks(np.arange(85, 96, 5))
# axes[0].grid(True)
# axes[0].legend()

# # ------------------------
# # Subplot 2 — 660k dataset
# # ------------------------
# axes[1].plot(df_66_size['epoch'], df_66_size['train_acc'], color='green', linewidth=2, alpha=0.7, label='Train Acc 660k')
# axes[1].plot(df_66_size['epoch'], df_66_size['val_acc'],   color='red',   linewidth=2, alpha=0.7, label='Val Acc 660k')
# axes[1].set_ylabel("Accuracy [%]")
# axes[1].set_title("660k Dataset - Model 2")
# axes[1].set_yticks(np.arange(85, 96, 5))
# axes[1].grid(True)
# axes[1].legend()

# # ------------------------
# # Subplot 3 — 330k dataset
# # ------------------------
# axes[2].plot(df_33_size['epoch'], df_33_size['train_acc'], color='purple', linewidth=2, alpha=0.7, label='Train Acc 330k')
# axes[2].plot(df_33_size['epoch'], df_33_size['val_acc'],   color='brown',  linewidth=2, alpha=0.7, label='Val Acc 330k')
# axes[2].set_ylabel("Accuracy [%]")
# axes[2].set_title("330k Dataset - Model 2")
# axes[2].set_yticks(np.arange(85, 96, 5))
# axes[2].set_xlabel("Epochs")
# axes[2].grid(True)
# axes[2].legend()

# # Final layout adjustments
# plt.tight_layout()
# plt.show()
# # --------------------------------------------------------------------------------------------------------------------

# # ----------------------------------- Inference Throughput/Delay/Jitter Plots -----------------------------------
# df = pd.read_csv("~/ns-3-dev/simulation_results.csv")
# df.columns = df.columns.str.strip()
# df['FlowType'] = (
#     df['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )

# # Group and compute mean throughput
# df_grouped = (
#     df.groupby(['scheduler', 'FlowType', 'ML'])['Jitter'] # Throughput/Delay/Jitter
#       .mean()
#       .reset_index()
# )

# def scheduler_label(row):
#     if row['scheduler'].endswith('RR'):
#         return 'RR'
#     elif row['scheduler'].endswith('PF'):
#         return 'PF'
#     elif row['scheduler'].endswith('Qos'):
#         if row['ML'] == 0:
#             return 'QoS'
#         else:
#             return "TRS"

# df_grouped['SchedulerLabel'] = df_grouped.apply(scheduler_label, axis=1)
# # print(df_grouped)
# print(df_grouped)
# flow_order = [
#     'UE1 Non-GBR Voice',
#     'UE2 GBR Video',
#     'UE3 DC-GBR',
#     'UE3 GBR Video',
#     'UE3 Non-GBR Voice'
# ]

# scheduler_order = ['RR', 'PF', 'QoS', 'TRS']

# # -------- 1 Plot for All Flows (Grouped Bars) --------
# # fig, ax = plt.subplots(figsize=(14, 6))

# # bar_width = 0.2
# # x = np.arange(len(flow_order))

# # colors = {
# #     'RR': 'tab:blue',
# #     'PF': 'tab:orange',
# #     'QoS': 'tab:green',
# #     'TRS': 'tab:red'
# # }

# # for i, scheduler in enumerate(scheduler_order):
# #     for j, flow in enumerate(flow_order):

# #         row = df_grouped[
# #             (df_grouped['FlowType'] == flow) &
# #             (df_grouped['SchedulerLabel'] == scheduler)
# #         ]

# #         if row.empty:
# #             continue  # do NOT plot fake zero bars

# #         ax.bar(
# #             x[j] + i * bar_width,
# #             row['Jitter'].values[0],
# #             width=bar_width,
# #             color=colors[scheduler],
# #             label=scheduler if j == 0 else None
# #         )

# # ax.set_xticks([i * bar_width for i in range(len(scheduler_order))])
# # ax.set_xticklabels(scheduler_order)


# # ax.set_xticks(x + bar_width * (len(scheduler_order) - 1) / 2)
# # ax.set_xticklabels(flow_order, rotation=20, ha='right')


# # ax.set_ylabel("Mean Jitter [ms]")
# # ax.set_xlabel("Flow Type")
# # ax.set_title("Mean Jitter per Flow and Scheduler")

# # ax.legend(title="Scheduler")
# # ax.grid(axis='y', linestyle='--', alpha=0.6)

# # plt.tight_layout()
# # plt.show()
# # --------------------------------------------------------

# # -------- Separate Plot per Flow (Single Bars) --------
# bar_width = 0.6  # wider bars since no grouping now

# colors = {
#     'RR': 'tab:blue',
#     'PF': 'tab:orange',
#     'QoS': 'tab:green',
#     'TRS': 'tab:red'
# }

# for flow in flow_order:

#     fig, ax = plt.subplots(figsize=(6, 4))

#     for i, scheduler in enumerate(scheduler_order):

#         row = df_grouped[
#             (df_grouped['FlowType'] == flow) &
#             (df_grouped['SchedulerLabel'] == scheduler)
#         ]

#         if row.empty:
#             continue

#         ax.bar(
#             i,
#             row['Jitter'].values[0],
#             width=bar_width,
#             color=colors[scheduler],
#             label=scheduler
#         )

#     ax.set_xticks(range(len(scheduler_order)))
#     ax.set_xticklabels(scheduler_order)

#     ax.set_ylabel("Mean Jitter [ms]")
#     ax.set_xlabel("Scheduler")
#     ax.set_title(f"Mean Jitter – {flow}")

#     ax.grid(axis='y', linestyle='--', alpha=0.6)

#     plt.tight_layout()
#     plt.show()
# # --------------------------------------------------------