import pandas as pd     
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# # --------------------------------------------------------------------------------------------------------------------
# # ---------------------------------- Model Training/Validation Accuracy Plots ----------------------------------
# # --------------------------------------------------------------------------------------------------------------------

# df_full_size = pd.read_csv("~/Masterarbeit/Results/Basic_model/training_validation_log_file_basic_full.csv")
# df_66_size = pd.read_csv("~/Masterarbeit/Results/Basic_model/training_validation_log_file_basic_66.csv")
# df_33_size = pd.read_csv("~/Masterarbeit/Results/Basic_model/training_validation_log_file_basic_33.csv")

# plt.plot(df_full_size['epoch'], df_full_size['train_acc'], color='blue', label='Dataset1: ~1M')
# plt.plot(df_66_size['epoch'], df_66_size['train_acc'], color='orange', label='Dataset2: ~660k')
# plt.plot(df_33_size['epoch'], df_33_size['train_acc'], color='green', label='Dataset3: ~330k')
# plt.plot(df_full_size['epoch'], df_full_size['val_acc'], color='blue', label='Dataset1: ~1M')
# plt.plot(df_66_size['epoch'], df_66_size['val_acc'], color='orange', label='Dataset2: ~660k')
# plt.plot(df_33_size['epoch'], df_33_size['val_acc'], color='green', label='Dataset3: ~330k')
# plt.plot(df_full_size['epoch'], df_full_size['train_acc'], color='blue', label='Train Acc 1M')
# plt.plot(df_full_size['epoch'], df_full_size['val_acc'], color='orange', label='Val Acc 1M')
# plt.plot(df_66_size['epoch'], df_66_size['train_acc'], color='green', label='Train Acc 660k')
# plt.plot(df_66_size['epoch'], df_66_size['val_acc'], color='red', label='Val Acc 660k')
# plt.plot(df_33_size['epoch'], df_33_size['train_acc'], color='purple', label='Train Acc 330k')
# plt.plot(df_33_size['epoch'], df_33_size['val_acc'], color='brown', label='Val Acc 330k')
# plt.yticks(np.arange(80, 105, 5))
# plt.xlabel("Epochs")
# plt.ylabel("Training/Validation Accuracy [%]")
# plt.title("Training/Validation Accuracy over Epochs for Dataset2 (~660k) - Model 1")
# plt.grid(True)
# plt.legend()
# plt.show()


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





# # ----------------------------------- Inference Plots network performance -----------------------------------
# df_traditional = pd.read_csv("~/ns-3-dev/simulation_results_traditional_QOS.csv")
# df_traditional.columns = df_traditional.columns.str.strip()
# df_traditional['FlowType'] = (
#     df_traditional['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )

# df_baseline = pd.read_csv("~/ns-3-dev/simulation_results.csv")
# df_baseline.columns = df_baseline.columns.str.strip()
# df_baseline['FlowType'] = (
#     df_baseline['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )
# df_baseline['scheduler'] = df_baseline['scheduler'].str.replace('ns3::NrMacSchedulerTdmaQos', 'MLP', regex=False)

# df_transformer = pd.read_csv("~/ns-3-dev/simulation_results_transformer_QOS.csv")
# df_transformer.columns = df_transformer.columns.str.strip()
# df_transformer['FlowType'] = (
#     df_transformer['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )

# df = pd.concat([df_traditional, df_baseline, df_transformer], ignore_index=True)

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
#     elif row['scheduler'].endswith('MLP'):
#         return "MLP"

# df['SchedulerLabel'] = df.apply(scheduler_label, axis=1)

# scheduler_order = ['QoS', 'TRS', 'MLP']


# metrics = ['Throughput', 'Delay', 'Jitter']
# ylabels = {
#     'Throughput': 'Average Throughput [Mbps]',
#     'Delay': 'Average Delay [ms]',
#     'Jitter': 'Average Jitter [ms]'
# }

# colors = {
#     'QoS': 'tab:green',
#     'TRS': 'tab:red',
#     'MLP': 'tab:purple'
# }

# df_plot = (
#     df
#     .groupby('SchedulerLabel')[metrics]
#     .mean()
#     .reindex(scheduler_order)
# )
# print(df_plot)

# fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

# x = np.arange(len(scheduler_order))

# for ax, metric in zip(axes, metrics):

#     values = df_plot[metric]

#     bars = ax.bar(
#         x,
#         values,
#         color=[colors[s] for s in scheduler_order]
#     )

#     ax.set_title(metric)
#     ax.set_ylabel(ylabels[metric])
#     ax.set_xticks(x)
#     ax.set_xticklabels(scheduler_order)
    
#     # Value labels
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(
#             f"{height:.2f}",
#             (bar.get_x() + bar.get_width() / 2, height),
#             ha='center', va='bottom', fontsize=9
#         )

#     ax.grid(axis='y', linestyle='--', alpha=0.6)

# legend_handles = [
#     Patch(facecolor=colors[s], label=s)
#     for s in scheduler_order
# ]

# fig.legend(
#     handles=legend_handles,
#     title="Scheduler",
#     loc="upper right",
#     bbox_to_anchor=(0.98, 0.98)
# )

# fig.suptitle("Average Network Performance per Scheduler", fontsize=14)
# plt.tight_layout()
# plt.show()
# # ---------------------------------------------------------------------------------------





# # ----------------------------------- Inference Plots performance per FlowType -----------------------------------
# df_traditional = pd.read_csv("~/ns-3-dev/simulation_results_traditional_QOS.csv")
# df_traditional.columns = df_traditional.columns.str.strip()
# df_traditional['FlowType'] = (
#     df_traditional['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )

# df_baseline = pd.read_csv("~/ns-3-dev/simulation_results_baseline_QOS.csv")
# df_baseline.columns = df_baseline.columns.str.strip()
# df_baseline['FlowType'] = (
#     df_baseline['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )
# df_baseline['scheduler'] = df_baseline['scheduler'].str.replace('ns3::NrMacSchedulerTdmaQos', 'MLP', regex=False)

# df_transformer = pd.read_csv("~/ns-3-dev/simulation_results_transformer_QOS.csv")
# df_transformer.columns = df_transformer.columns.str.strip()
# df_transformer['FlowType'] = (
#     df_transformer['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )

# df = pd.concat([df_traditional, df_baseline, df_transformer], ignore_index=True)

# df_grouped = (
#     df.groupby(['scheduler', 'FlowType', 'ML'])[['Throughput', 'Delay', 'Jitter']].mean().reset_index()
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
#     elif row['scheduler'].endswith('MLP'):
#         return "MLP"

# df_grouped['SchedulerLabel'] = df_grouped.apply(scheduler_label, axis=1)

# scheduler_order = ['QoS', 'TRS', 'MLP']

# flow_order = [
#     'UE1 Non-GBR Voice',
#     'UE2 GBR Video',
#     'UE3 DC-GBR',
#     'UE3 GBR Video',
#     'UE3 Non-GBR Voice'
# ]

# metrics = ['Throughput', 'Delay', 'Jitter']
# ylabels = {
#     'Throughput': 'Average Throughput [Mbps]',
#     'Delay': 'Average Delay [ms]',
#     'Jitter': 'Average Jitter [ms]'
# }

# colors = {
#     'QoS': 'tab:green',
#     'TRS': 'tab:red',
#     'MLP': 'tab:purple'
# }

# x = np.arange(len(scheduler_order))

# for flow in flow_order:

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

#     for ax, metric in zip(axes, metrics):

#         data = (
#             df_grouped[df_grouped['FlowType'] == flow]
#             .set_index('SchedulerLabel')
#             .reindex(scheduler_order)
#         )

#         values = data[metric]

#         bars = ax.bar(
#             x,
#             values,
#             color=[colors[s] for s in scheduler_order]
#         )

#         ax.set_title(metric)
#         ax.set_ylabel(ylabels[metric])
#         ax.set_xticks(x)
#         ax.set_xticklabels(scheduler_order)

#         # # Optional log scale for delay & jitter
#         # if metric in ['Delay', 'Jitter']:
#         #     ax.set_yscale('log')

#         # Value labels
#         for bar in bars:
#             height = bar.get_height()
#             ax.annotate(
#                 f"{height:.2f}",
#                 (bar.get_x() + bar.get_width() / 2, height),
#                 ha='center', va='bottom', fontsize=9
#             )

#         ax.grid(axis='y', linestyle='--', alpha=0.6)

#     # Global legend (once per figure)
#     legend_handles = [
#         Patch(facecolor=colors[s], label=s)
#         for s in scheduler_order
#     ]

#     fig.legend(
#         handles=legend_handles,
#         title="Scheduler",
#         loc="upper right",
#         bbox_to_anchor=(0.98, 0.98)
#     )

#     fig.suptitle(f"Average Performance of Flow per Scheduler – {flow}", fontsize=14)
#     plt.tight_layout()
#     plt.show()
# # ---------------------------------------------------------------------------------------





# # ----------------------------------- Inference Transformer only -----------------------------------
# df_transformer = pd.read_csv("~/ns-3-dev/simulation_results_transformer_QOS.csv")
# df_transformer.columns = df_transformer.columns.str.strip()
# df_transformer['FlowType'] = (
#     df_transformer['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )

# flow_order = [
#     'UE1 Non-GBR Voice',
#     'UE2 GBR Video',
#     'UE3 DC-GBR',
#     'UE3 GBR Video',
#     'UE3 Non-GBR Voice'
# ]

# metrics = ['Throughput', 'Delay', 'Jitter']
# ylabels = {
#     'Throughput': 'Average Throughput [Mbps]',
#     'Delay': 'Average Delay [ms]',
#     'Jitter': 'Average Jitter [ms]'
# }

# colors = {
#     'UE1 Non-GBR Voice': 'tab:green',
#     'UE2 GBR Video': 'tab:blue',
#     'UE3 DC-GBR': 'tab:orange',
#     'UE3 GBR Video': 'tab:red',
#     'UE3 Non-GBR Voice': 'tab:purple'
# }

# df_plot = (
#     df_transformer
#     .groupby('FlowType')[metrics]
#     .mean()
#     .reindex(flow_order)
# )
# print(df_plot)

# fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

# x = np.arange(len(flow_order))

# for ax, metric in zip(axes, metrics):

#     values = df_plot[metric]

#     bars = ax.bar(
#         x,
#         values,
#         color=[colors[s] for s in flow_order]
#     )

#     ax.set_title(metric)
#     ax.set_ylabel(ylabels[metric])
#     ax.set_xticks(x)
#     ax.set_xticklabels(flow_order)
    
#     # Value labels
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(
#             f"{height:.2f}",
#             (bar.get_x() + bar.get_width() / 2, height),
#             ha='center', va='bottom', fontsize=9
#         )

#     ax.grid(axis='y', linestyle='--', alpha=0.6)

# legend_handles = [
#     Patch(facecolor=colors[s], label=s)
#     for s in flow_order
# ]

# fig.legend(
#     handles=legend_handles,
#     title="Scheduler",
#     loc="upper right",
#     bbox_to_anchor=(0.98, 0.98)
# )

# fig.suptitle("Average Network Performance per Scheduler", fontsize=14)
# plt.tight_layout()
# plt.show()
# # ---------------------------------------------------------------------------------------





# # ---------------------------------- Plots for learning rate influence on training -------------------------------------
# df_lr1e3 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e3_bs32.csv")
# df_lr1e4 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 
# df_lr1e5 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e5_bs32.csv")

# def plot_with_max(ax, df, num_layers, d_model, nhead, color, lr_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     ax.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=2,
#         alpha=0.7,
#         label=f"{lr_label} (max {max_acc:.2f}% @ {int(max_epoch)})"
#     )

#     ax.scatter(max_epoch, max_acc, edgecolor='black', color=color, zorder=20)

# plt.rcParams.update({
#     "font.size": 10,
#     "font.family": "serif",
#     "axes.titlesize": 10,
#     "axes.labelsize": 10,
#     "legend.fontsize": 11
# })
# fig, axes = plt.subplots(4, 1, figsize=(6.1, 8), constrained_layout=True, sharex=True)

# # ----------------------
# # Subplot 1 — Model 1 - d_model=16, num_layers=1, num_heads=4, parameters: 79.3k
# # ----------------------
# plot_with_max(axes[0], df_lr1e3, num_layers=1, d_model=16, nhead=4, color='blue', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-3')
# plot_with_max(axes[0], df_lr1e4, num_layers=1, d_model=16, nhead=4, color='orange', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-4')
# plot_with_max(axes[0], df_lr1e5, num_layers=1, d_model=16, nhead=4, color='green', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-5')
# axes[0].set_ylabel("Validation Accuracy [%]")
# axes[0].set_title("Model 1 - d_model=16, num_layers=1, num_heads=4, parameters: 79.3k")
# axes[0].set_yticks(np.arange(60, 80, 5))
# axes[0].grid(True)
# axes[0].legend(loc= 'lower right')

# # ------------------------
# # Subplot 2 — Model 2 - d_model=16, num_layers=5, num_heads=4, parameters: 354.3k"
# # ------------------------
# plot_with_max(axes[1], df_lr1e3, num_layers=5, d_model=16, nhead=4, color='blue', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-3')
# plot_with_max(axes[1], df_lr1e4, num_layers=5, d_model=16, nhead=4, color='orange', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-4')
# plot_with_max(axes[1], df_lr1e5, num_layers=5, d_model=16, nhead=4, color='green', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-5')
# axes[1].set_ylabel("Validation Accuracy [%]")
# axes[1].set_title("Model 2 - d_model=16, num_layers=5, num_heads=4, parameters: 354.3k")
# axes[1].set_yticks(np.arange(60, 80, 5))
# axes[1].grid(True)
# axes[1].legend(loc= 'lower right')

# # ------------------------
# # Subplot 3 — Model 3 - d_model=64, num_layers=3, num_heads=4, parameters: 916.2k
# # ------------------------
# plot_with_max(axes[2], df_lr1e3, num_layers=3, d_model=64, nhead=4, color='blue', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-3')
# plot_with_max(axes[2], df_lr1e4, num_layers=3, d_model=64, nhead=4, color='orange', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-4')
# plot_with_max(axes[2], df_lr1e5, num_layers=3, d_model=64, nhead=4, color='green', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-5')
# axes[2].set_ylabel("Validation Accuracy [%]")
# axes[2].set_title("Model 3 - d_model=64, num_layers=3, num_heads=4, parameters: 916.2k")
# axes[2].set_yticks(np.arange(60, 80, 5))
# axes[2].grid(True)
# axes[2].legend(loc= 'lower right')

# # ------------------------
# # Subplot 4 — Model 4 - d_model=128, num_layers=6, num_heads=4, parameters: 3.8M
# # ------------------------
# plot_with_max(axes[3], df_lr1e3, num_layers=6, d_model=128, nhead=4, color='blue', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-3')
# plot_with_max(axes[3], df_lr1e4, num_layers=6, d_model=128, nhead=4, color='orange', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-4')
# plot_with_max(axes[3], df_lr1e5, num_layers=6, d_model=128, nhead=4, color='green', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-5')
# axes[3].set_ylabel("Validation Accuracy [%]")
# axes[3].set_title("Model 4 - d_model=128, num_layers=6, num_heads=4, parameters: 3.8M")
# axes[3].set_xlabel("Epochs")
# axes[3].grid(True)
# axes[3].legend(loc= 'lower right')

# # Final layout adjustments
# plt.tight_layout()
# plt.savefig("/home/maximilianrosca/Masterarbeit/training_dataset6/learning_rate_influence.pdf", bbox_inches='tight', pad_inches=0.02)
# plt.show()
# # ------------------------------- End Plots for learning rate influence on training----------------------------------------------





# # ---------------------------------- Plots for batch size influence on training -------------------------------------
# df_bs16 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_bs16-forplotsonly.csv")
# df_bs32 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 
# df_bs64 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs64.csv")

# def plot_with_max(ax, df, num_layers, d_model, nhead, lr, color, bs_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead) &
#         (df['lr'] == lr)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     ax.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=2,
#         alpha=0.7,
#         label=f"{bs_label} (max {max_acc:.2f}% @ {int(max_epoch)})"
#     )

#     ax.scatter(max_epoch, max_acc, edgecolor='black', color=color, zorder=20)

# plt.rcParams.update({
#     "font.size": 10,
#     "font.family": "serif",
#     "axes.titlesize": 10,
#     "axes.labelsize": 10,
#     "legend.fontsize": 11
# })

# fig, axes = plt.subplots(4, 1, figsize=(6.1, 8), constrained_layout=True, sharex=True)

# # ----------------------
# # Subplot 1 — Model 1 - d_model=16, num_layers=1, num_heads=4, parameters: 79.3k"
# # ----------------------
# plot_with_max(axes[0], df_bs16, num_layers=1, d_model=16, nhead=4, lr=1e-4, color='blue', bs_label='b_size: 16')
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=4, lr=1e-4, color='orange', bs_label='b_size: 32')
# plot_with_max(axes[0], df_bs64, num_layers=1, d_model=16, nhead=4, lr=1e-4, color='green', bs_label='b_size: 64')
# axes[0].set_ylabel("Validation Accuracy [%]")
# axes[0].set_title("Model 1 - d_model=16, num_layers=1, num_heads=4, parameters: 79.3k")
# axes[0].set_yticks(np.arange(68, 80, 2))
# axes[0].grid(True)
# axes[0].legend(loc= "lower right")

# # ------------------------
# # Subplot 2 — Model 2 - d_model=16, num_layers=5, num_heads=4, parameters: 354.3k
# # ------------------------
# plot_with_max(axes[1], df_bs16, num_layers=5, d_model=16, nhead=4, lr=1e-4, color='blue', bs_label='b_size: 16')
# plot_with_max(axes[1], df_bs32, num_layers=5, d_model=16, nhead=4, lr=1e-4, color='orange', bs_label='b_size: 32')
# plot_with_max(axes[1], df_bs64, num_layers=5, d_model=16, nhead=4, lr=1e-4, color='green', bs_label='b_size: 64')
# axes[1].set_ylabel("Validation Accuracy [%]")
# axes[1].set_title("Model 2 - d_model=16, num_layers=5, num_heads=4, parameters: 354.3k")
# axes[1].set_yticks(np.arange(68, 80, 2))
# axes[1].grid(True)
# axes[1].legend(loc= "lower right")

# # ------------------------
# # Subplot 3 — Model 3 - d_model=64, num_layers=3, num_heads=4, parameters: 916.2k
# # ------------------------
# plot_with_max(axes[2], df_bs16, num_layers=3, d_model=64, nhead=4, lr=1e-4, color='blue', bs_label='b_size: 16')
# plot_with_max(axes[2], df_bs32, num_layers=3, d_model=64, nhead=4, lr=1e-4, color='orange', bs_label='b_size: 32')
# plot_with_max(axes[2], df_bs64, num_layers=3, d_model=64, nhead=4, lr=1e-4, color='green', bs_label='b_size: 64')
# axes[2].set_ylabel("Validation Accuracy [%]")
# axes[2].set_title("Model 3 - d_model=64, num_layers=3, num_heads=4, parameters: 916.2k")
# axes[2].set_yticks(np.arange(68, 80, 2))
# axes[2].grid(True)
# axes[2].legend(loc= "lower right")

# # ------------------------
# # Subplot 4 — Model 4 - d_model=128, num_layers=6, num_heads=4, parameters: 3.8M
# # ------------------------
# plot_with_max(axes[3], df_bs16, num_layers=6, d_model=128, nhead=4, lr=1e-4, color='blue', bs_label='b_size: 16')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=4, lr=1e-4, color='orange', bs_label='b_size: 32')
# plot_with_max(axes[3], df_bs64, num_layers=6, d_model=128, nhead=4, lr=1e-4, color='green', bs_label='b_size: 64')
# axes[3].set_ylabel("Validation Accuracy [%]")
# axes[3].set_title("Model 4 - d_model=128, num_layers=6, num_heads=4, parameters: 3.8M")
# axes[3].set_yticks(np.arange(68, 80, 2))
# axes[3].set_xlabel("Epochs")
# axes[3].grid(True)
# axes[3].legend(loc= "lower right")

# # Final layout adjustments
# plt.tight_layout()
# plt.savefig("/home/maximilianrosca/Masterarbeit/training_dataset6/batch_size_influence.pdf", bbox_inches='tight', pad_inches=0.02)
# plt.show()
# # ------------------------------- End Plots for batch size influence on training----------------------------------------------





# # ---------------------------------- Plots for number of encoder layers influence on training -------------------------------------
# df_bs32 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 

# def plot_with_max(ax, df, num_layers, d_model, nhead, lr, color, num_layers_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead) &
#         (df['lr'] == lr)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     ax.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=2,
#         alpha=0.7,
#         label=f"{num_layers_label} (max {max_acc:.2f}% @ {int(max_epoch)})"
#     )

#     ax.scatter(max_epoch, max_acc, edgecolor='black', color=color, zorder=20)

# plt.rcParams.update({
#     "font.size": 10,
#     "font.family": "serif",
#     "axes.titlesize": 10,
#     "axes.labelsize": 10,
#     "legend.fontsize": 8
# })
# fig, axes = plt.subplots(4, 1, figsize=(6.1, 8), constrained_layout=True, sharex=True)

# # ----------------------
# # Subplot 1 — Model 1 (small) - d_model=16, num_heads=4, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=4, lr=1e-4, color='blue', num_layers_label='Encoder layers: 1')
# plot_with_max(axes[0], df_bs32, num_layers=2, d_model=16, nhead=4, lr=1e-4, color='orange', num_layers_label='Encoder layers: 2')
# plot_with_max(axes[0], df_bs32, num_layers=3, d_model=16, nhead=4, lr=1e-4, color='green', num_layers_label='Encoder layers: 3')
# plot_with_max(axes[0], df_bs32, num_layers=4, d_model=16, nhead=4, lr=1e-4, color='red', num_layers_label='Encoder layers: 4')
# plot_with_max(axes[0], df_bs32, num_layers=5, d_model=16, nhead=4, lr=1e-4, color='purple', num_layers_label='Encoder layers: 5')
# plot_with_max(axes[0], df_bs32, num_layers=6, d_model=16, nhead=4, lr=1e-4, color='brown', num_layers_label='Encoder layers: 6')
# axes[0].set_ylabel("Validation Accuracy [%]")
# axes[0].set_title("Model 1 - d_model=16, num_heads=4, learning rate=1e-4, batch size=32")
# axes[0].set_yticks(np.arange(68, 80, 2))
# axes[0].grid(True)
# axes[0].legend(loc= "lower right")

# # ----------------------
# # Subplot 2 — Model 2 (small-to-medium) - d_model=32, num_heads=4, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[1], df_bs32, num_layers=1, d_model=32, nhead=4, lr=1e-4, color='blue', num_layers_label='Encoder layers: 1')
# plot_with_max(axes[1], df_bs32, num_layers=2, d_model=32, nhead=4, lr=1e-4, color='orange', num_layers_label='Encoder layers: 2')
# plot_with_max(axes[1], df_bs32, num_layers=3, d_model=32, nhead=4, lr=1e-4, color='green', num_layers_label='Encoder layers: 3')
# plot_with_max(axes[1], df_bs32, num_layers=4, d_model=32, nhead=4, lr=1e-4, color='red', num_layers_label='Encoder layers: 4')
# plot_with_max(axes[1], df_bs32, num_layers=5, d_model=32, nhead=4, lr=1e-4, color='purple', num_layers_label='Encoder layers: 5')
# plot_with_max(axes[1], df_bs32, num_layers=6, d_model=32, nhead=4, lr=1e-4, color='brown', num_layers_label='Encoder layers: 6')
# axes[1].set_ylabel("Validation Accuracy [%]")
# axes[1].set_title("Model 2 - d_model=32, num_heads=4, learning rate=1e-4, batch size=32")
# axes[1].set_yticks(np.arange(68, 80, 2))
# axes[1].grid(True)
# axes[1].legend(loc= "lower right")

# # ----------------------
# # Subplot 3 — Model 3 (medium-to-large) - d_model=64, num_heads=4, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[2], df_bs32, num_layers=1, d_model=64, nhead=4, lr=1e-4, color='blue', num_layers_label='Encoder layers: 1')
# plot_with_max(axes[2], df_bs32, num_layers=2, d_model=64, nhead=4, lr=1e-4, color='orange', num_layers_label='Encoder layers: 2')
# plot_with_max(axes[2], df_bs32, num_layers=3, d_model=64, nhead=4, lr=1e-4, color='green', num_layers_label='Encoder layers: 3')
# plot_with_max(axes[2], df_bs32, num_layers=4, d_model=64, nhead=4, lr=1e-4, color='red', num_layers_label='Encoder layers: 4')
# plot_with_max(axes[2], df_bs32, num_layers=5, d_model=64, nhead=4, lr=1e-4, color='purple', num_layers_label='Encoder layers: 5')
# plot_with_max(axes[2], df_bs32, num_layers=6, d_model=64, nhead=4, lr=1e-4, color='brown', num_layers_label='Encoder layers: 6')
# axes[2].set_ylabel("Validation Accuracy [%]")
# axes[2].set_title("Model 3 - d_model=64, num_heads=4, learning rate=1e-4, batch size=32")
# axes[2].set_yticks(np.arange(68, 80, 2))
# axes[2].grid(True)
# axes[2].legend(loc= "lower right")

# # ----------------------
# # Subplot 4 — Model 3 (large) - d_model=128, num_heads=4, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[3], df_bs32, num_layers=1, d_model=128, nhead=4, lr=1e-4, color='blue', num_layers_label='Encoder layers: 1')
# plot_with_max(axes[3], df_bs32, num_layers=2, d_model=128, nhead=4, lr=1e-4, color='orange', num_layers_label='Encoder layers: 2')
# plot_with_max(axes[3], df_bs32, num_layers=3, d_model=128, nhead=4, lr=1e-4, color='green', num_layers_label='Encoder layers: 3')
# plot_with_max(axes[3], df_bs32, num_layers=4, d_model=128, nhead=4, lr=1e-4, color='red', num_layers_label='Encoder layers: 4')
# plot_with_max(axes[3], df_bs32, num_layers=5, d_model=128, nhead=4, lr=1e-4, color='purple', num_layers_label='Encoder layers: 5')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=4, lr=1e-4, color='brown', num_layers_label='Encoder layers: 6')
# axes[3].set_ylabel("Validation Accuracy [%]")
# axes[3].set_title("Model 4 - d_model=128, num_heads=4, learning rate=1e-4, batch size=32")
# axes[3].set_yticks(np.arange(68, 80, 2))
# axes[3].grid(True)
# axes[3].legend(loc= "lower right")
# axes[3].set_xlabel("Epochs")

# # Final layout adjustments
# plt.tight_layout()
# plt.savefig("/home/maximilianrosca/Masterarbeit/training_dataset6/encoder_layers_influence.pdf", bbox_inches='tight', pad_inches=0.02)
# plt.show()
# # ---------------------------------- End of Plots for number of encoder layers influence on training -------------------------------------





# # ---------------------------------- Plots for number of attention heads influence on training -------------------------------------
# df_bs32 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 

# def plot_with_max(df, num_layers, d_model, nhead, lr, color, num_heads_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead) &
#         (df['lr'] == lr)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     plt.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=2,
#         alpha=0.7,
#         label=f"{num_heads_label} (max {max_acc:.2f}% @ {int(max_epoch)})"
#     )

#     plt.scatter(max_epoch, max_acc, edgecolor='black', color=color, zorder=20)

# plot_with_max(df_bs32, num_layers=3, d_model=32, nhead=1, lr=1e-4, color='blue', num_heads_label='Attention heads: 1')
# plot_with_max(df_bs32, num_layers=3, d_model=32, nhead=2, lr=1e-4, color='orange', num_heads_label='Attention heads: 2')
# plot_with_max(df_bs32, num_layers=3, d_model=32, nhead=4, lr=1e-4, color='green', num_heads_label='Attention heads: 4')
# plot_with_max(df_bs32, num_layers=3, d_model=32, nhead=8, lr=1e-4, color='red', num_heads_label='Attention heads: 8')
# # plt.yticks(np.arange(68, 80, 2))
# plt.xlabel("Epochs")
# plt.ylabel("Validation Accuracy [%]")
# plt.title("Model - d_model=32, num_layers=3, batch_size=32, lr=1e-4, epochs=300, dataset=346.866 samples")
# plt.grid(True)
# plt.legend(loc= "lower right")
# plt.show()
# # ---------------------------------- End of Plots for number of attention heads influence on training -------------------------------------





# # ---------------------------------- Plots for number of attention heads influence on training - subplots -------------------------------------
# df_bs32 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 

# def plot_with_max(ax, df, num_layers, d_model, nhead, lr, color, num_heads_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead) &
#         (df['lr'] == lr)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     ax.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=2,
#         alpha=0.7,
#         label=f"{num_heads_label} (max {max_acc:.2f}% @ {int(max_epoch)})"
#     )

#     ax.scatter(max_epoch, max_acc, edgecolor='black', color=color, zorder=20)

# plt.rcParams.update({
#     "font.size": 10,
#     "font.family": "serif",
#     "axes.titlesize": 10,
#     "axes.labelsize": 10,
#     "legend.fontsize": 8
# })
# fig, axes = plt.subplots(4, 1, figsize=(6.1, 8), constrained_layout=True, sharex=True)

# # ----------------------
# # Subplot 1 — Model 1 (small) - d_model=16, num_layers=1, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=1, lr=1e-4, color='blue', num_heads_label='Attention heads: 1')
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=2, lr=1e-4, color='orange', num_heads_label='Attention heads: 2')
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=4, lr=1e-4, color='green', num_heads_label='Attention heads: 4')
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=8, lr=1e-4, color='red', num_heads_label='Attention heads: 8')

# axes[0].set_ylabel("Validation Accuracy [%]")
# axes[0].set_title("Model 1 - d_model=16, num_layers=1, parameters: 79.3k")
# axes[0].set_yticks(np.arange(68, 80, 2))
# axes[0].grid(True)
# axes[0].legend(loc= "lower right")

# # ----------------------
# # Subplot 2 — Model 2 (small-to-medium) - d_model=16, num_layers=5, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[1], df_bs32, num_layers=5, d_model=16, nhead=1, lr=1e-4, color='blue', num_heads_label='Attention heads: 1')
# plot_with_max(axes[1], df_bs32, num_layers=5, d_model=16, nhead=2, lr=1e-4, color='orange', num_heads_label='Attention heads: 2')
# plot_with_max(axes[1], df_bs32, num_layers=5, d_model=16, nhead=4, lr=1e-4, color='green', num_heads_label='Attention heads: 4')
# plot_with_max(axes[1], df_bs32, num_layers=5, d_model=16, nhead=8, lr=1e-4, color='red', num_heads_label='Attention heads: 8')
# axes[1].set_ylabel("Validation Accuracy [%]")
# axes[1].set_title("Model 2 - d_model=16, num_layers=5, parameters: 354.3k")
# axes[1].set_yticks(np.arange(68, 80, 2))
# axes[1].grid(True)
# axes[1].legend(loc= "lower right")

# # ----------------------
# # Subplot 3 — Model 3 (medium-to-large) - d_model=64, num_layers=3, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[2], df_bs32, num_layers=3, d_model=64, nhead=1, lr=1e-4, color='blue', num_heads_label='Attention heads: 1')
# plot_with_max(axes[2], df_bs32, num_layers=3, d_model=64, nhead=2, lr=1e-4, color='orange', num_heads_label='Attention heads: 2')
# plot_with_max(axes[2], df_bs32, num_layers=3, d_model=64, nhead=4, lr=1e-4, color='green', num_heads_label='Attention heads: 4')
# plot_with_max(axes[2], df_bs32, num_layers=3, d_model=64, nhead=8, lr=1e-4, color='red', num_heads_label='Attention heads: 8')
# axes[2].set_ylabel("Validation Accuracy [%]")
# axes[2].set_title("Model 3 - d_model=64, num_layers=3, parameters: 916.2k")
# axes[2].set_yticks(np.arange(68, 80, 2))
# axes[2].grid(True)
# axes[2].legend(loc= "lower right")

# # ----------------------
# # Subplot 4 — Model 3 (large) - d_model=128, num_heads=6, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=1, lr=1e-4, color='blue', num_heads_label='Attention heads: 1')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=2, lr=1e-4, color='orange', num_heads_label='Attention heads: 2')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=4, lr=1e-4, color='green', num_heads_label='Attention heads: 4')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=8, lr=1e-4, color='red', num_heads_label='Attention heads: 8')
# axes[3].set_ylabel("Validation Accuracy [%]")
# axes[3].set_title("Model 4 - d_model=128, num_layers=6, parameters: 3.8M")
# axes[3].set_yticks(np.arange(68, 80, 2))
# axes[3].grid(True)
# axes[3].legend(loc= "lower right")
# axes[3].set_xlabel("Epochs")

# # Final layout adjustments
# plt.tight_layout()
# plt.savefig("/home/maximilianrosca/Masterarbeit/training_dataset6/attention_heads_influence.pdf", bbox_inches='tight', pad_inches=0.02)
# plt.show()
# # ---------------------------------- End of Plots for number of attention heads influence on training - subplots -------------------------------------





# # ---------------------------------- Plots for model dimension influence on training -------------------------------------
# df_bs32 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 

# def plot_with_max(df, num_layers, d_model, nhead, lr, color, d_model_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead) &
#         (df['lr'] == lr)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     plt.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=2,
#         alpha=0.7,
#         label=f"{d_model_label} (max {max_acc:.2f}% @ {int(max_epoch)})"
#     )

#     plt.scatter(max_epoch, max_acc, color='black', zorder=20)

# plot_with_max(df_bs32, num_layers=3, d_model=16, nhead=4, lr=1e-4, color='blue', d_model_label='Model dimension: 16')
# plot_with_max(df_bs32, num_layers=3, d_model=32, nhead=4, lr=1e-4, color='orange', d_model_label='Model dimension: 32')
# plot_with_max(df_bs32, num_layers=3, d_model=64, nhead=4, lr=1e-4, color='green', d_model_label='Model dimension: 64')
# plot_with_max(df_bs32, num_layers=3, d_model=128, nhead=4, lr=1e-4, color='red', d_model_label='Model dimension: 128')
# # plt.yticks(np.arange(68, 80, 2))
# plt.xlabel("Epochs")
# plt.ylabel("Validation Accuracy [%]")
# plt.title("Model - num_layers=3, nheads=4, batch_size=32, lr=1e-4, epochs=300, dataset=346.866 samples")
# plt.grid(True)
# plt.legend(loc= "lower right")
# plt.show()
# # ---------------------------------- End of Plots for model dimension influence on training -------------------------------------





# # ---------------------------------- Plots for model dimension influence on training - subplots -------------------------------------
# df_bs32 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 

# def plot_with_max(ax, df, num_layers, d_model, nhead, lr, color, d_model_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead) &
#         (df['lr'] == lr)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     ax.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=2,
#         alpha=0.7,
#         label=f"{d_model_label} (max {max_acc:.2f}% @ {int(max_epoch)})"
#     )

#     ax.scatter(max_epoch, max_acc, edgecolor='black', color=color, zorder=20)

# plt.rcParams.update({
#     "font.size": 10,
#     "font.family": "serif",
#     "axes.titlesize": 10,
#     "axes.labelsize": 10,
#     "legend.fontsize": 8
# })

# fig, axes = plt.subplots(4, 1, figsize=(6.1, 8), constrained_layout=True, sharex=True)

# # ----------------------
# # Subplot 1 — Model 1 (small) - num_layers=1, nheads=4, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=4, lr=1e-4, color='blue', d_model_label='Model dimension: 16')
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=32, nhead=4, lr=1e-4, color='orange', d_model_label='Model dimension: 32')
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=64, nhead=4, lr=1e-4, color='green', d_model_label='Model dimension: 64')
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=128, nhead=4, lr=1e-4, color='red', d_model_label='Model dimension: 128')

# axes[0].set_ylabel("Validation Accuracy [%]")
# axes[0].set_title("Model 1 - num_layers=1, nheads=4, learning rate=1e-4, batch_size=32")
# axes[0].set_yticks(np.arange(68, 80, 2))
# axes[0].grid(True)
# axes[0].legend(loc= "lower right")

# # ----------------------
# # Subplot 2 — Model 2 (small-to-medium) - num_layers=3, nheads=4, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[1], df_bs32, num_layers=3, d_model=16, nhead=4, lr=1e-4, color='blue', d_model_label='Model dimension: 16')
# plot_with_max(axes[1], df_bs32, num_layers=3, d_model=32, nhead=4, lr=1e-4, color='orange', d_model_label='Model dimension: 32')
# plot_with_max(axes[1], df_bs32, num_layers=3, d_model=64, nhead=4, lr=1e-4, color='green', d_model_label='Model dimension: 64')
# plot_with_max(axes[1], df_bs32, num_layers=3, d_model=128, nhead=4, lr=1e-4, color='red', d_model_label='Model dimension: 128')
# axes[1].set_ylabel("Validation Accuracy [%]")
# axes[1].set_title("Model 2 - num_layers=3, nheads=4, learning rate=1e-4, batch_size=32")
# axes[1].set_yticks(np.arange(68, 80, 2))
# axes[1].grid(True)
# axes[1].legend(loc= "lower right")

# # ----------------------
# # Subplot 3 — Model 3 (medium-to-large) - num_layers=4, nheads=4, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[2], df_bs32, num_layers=4, d_model=16, nhead=4, lr=1e-4, color='blue', d_model_label='Model dimension: 16')
# plot_with_max(axes[2], df_bs32, num_layers=4, d_model=32, nhead=4, lr=1e-4, color='orange', d_model_label='Model dimension: 32')
# plot_with_max(axes[2], df_bs32, num_layers=4, d_model=64, nhead=4, lr=1e-4, color='green', d_model_label='Model dimension: 64')
# plot_with_max(axes[2], df_bs32, num_layers=4, d_model=128, nhead=4, lr=1e-4, color='red', d_model_label='Model dimension: 128')
# axes[2].set_ylabel("Validation Accuracy [%]")
# axes[2].set_title("Model 3 - num_layers=4, nheads=4, learning rate=1e-4, batch_size=32")
# axes[2].set_yticks(np.arange(68, 80, 2))
# axes[2].grid(True)
# axes[2].legend(loc= "lower right")

# # ----------------------
# # Subplot 4 — Model 4 (large) - num_layers=6, num_heads=4, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=16, nhead=4, lr=1e-4, color='blue', d_model_label='Model dimension: 16')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=32, nhead=4, lr=1e-4, color='orange', d_model_label='Model dimension: 32')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=64, nhead=4, lr=1e-4, color='green', d_model_label='Model dimension: 64')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=4, lr=1e-4, color='red', d_model_label='Model dimension: 128')
# axes[3].set_ylabel("Validation Accuracy [%]")
# axes[3].set_title("Model 4 - num_layers=6, nheads=4, learning rate=1e-4, batch_size=32")
# axes[3].set_yticks(np.arange(68, 80, 2))
# axes[3].grid(True)
# axes[3].legend(loc= "lower right")
# axes[3].set_xlabel("Epochs")

# # Final layout adjustments
# plt.tight_layout()
# plt.savefig("/home/maximilianrosca/Masterarbeit/training_dataset6/model_dimension_influence.pdf", bbox_inches='tight', pad_inches=0.02)
# plt.show()
# # ---------------------------------- End of Plots for model dimension influence on training - subplot -------------------------------------





# # --------------------------- Testing plots for paper (throughput, delay, jitter) -------------------------------------------
# df = pd.read_csv('/home/maximilianrosca/ns-3-dev/simulation_results_5flows_5mhz.csv')
# df.columns = df.columns.str.strip()
# df['FlowType'] = (
#     df['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )

# df_baseline = pd.read_csv('/home/maximilianrosca/ns-3-dev/simulation_results_5flows_5mhz_mlp.csv')
# df_baseline.columns = df_baseline.columns.str.strip()
# df_baseline['FlowType'] = (
#     df_baseline['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )
# df_baseline['scheduler'] = df_baseline['scheduler'].str.replace('ns3::NrMacSchedulerTdmaQos', 'MLP', regex=False)

# df = pd.concat([df, df_baseline], ignore_index=True)

# def scheduler_label(row):
#     if row['scheduler'].endswith('RR'):
#         return 'RR'
#     elif row['scheduler'].endswith('PF'):
#         return 'PF'
#     elif row['scheduler'].endswith('Qos'):
#         if row['ML'] == 0:
#             return 'QoS'
#         else:
#             return "TF QoS"
#     elif row['scheduler'] == 'MLP':
#         return 'MLP'

# df['SchedulerLabel'] = df.apply(scheduler_label, axis=1)

# flow_order = [
#     'UE1 Non-GBR',
#     'UE1 GBR',
#     'UE2 GBR',
#     'UE3 Non-GBR',
#     'UE3 DC-GBR'
# ]

# scheduler_order = ['RR', 'PF', 'QoS', 'TF QoS', 'MLP']

# colors = {
#     'RR': 'tab:blue',
#     'PF': 'tab:orange',
#     'QoS': 'tab:green',
#     'TF QoS': 'tab:red',
#     'MLP': 'tab:purple'
# }

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.size": 9,
#     "axes.titlesize": 9,
#     "axes.labelsize": 9,
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "legend.fontsize": 8,
#     "figure.titlesize": 9
# })

# max_throughput = {
#     'UE1 Non-GBR': 5,
#     'UE1 GBR': 10,
#     'UE2 GBR': 10,
#     'UE3 Non-GBR': 5,
#     'UE3 DC-GBR': 15
# }

# def plot_metric(metric, ylabel, num_flows, bandwidth, filename):

#     fig, ax = plt.subplots(figsize=(6.1, 2.8))

#     bar_width = 0.13
#     # x = np.arange(len(flow_order))
#     # x = np.array([0, 1, 2.3, 3.3])
#     x = np.array([0, 1, 2.4, 3.6, 4.6])


#     for i, scheduler in enumerate(scheduler_order):
#         for j, flow in enumerate(flow_order):

#             row = df[
#                 (df['FlowType'] == flow) &
#                 (df['SchedulerLabel'] == scheduler)
#             ]
            
#             MAX_DELAY = 800
#             if row[metric].values[0] == 0:
#                 value = MAX_DELAY
#                 is_missing = True
#             else:
#                 value = row[metric].values[0]
#                 is_missing = False
#             # value = row[metric].values[0]
#             # is_missing = False           

#             ax.bar(
#                 x[j] + i * bar_width,
#                 value,
#                 width=bar_width,
#                 color=colors[scheduler],
#                 edgecolor='black' if is_missing else None,
#                 hatch='//' if is_missing else None,
#                 label=scheduler if ((j == 0) and not is_missing) or (j == 2 and scheduler == 'MLP') else None
#                 # label = scheduler if (j == 0) else None
#             )

#             if is_missing:
#                 ax.text(
#                     x[j] + i * bar_width,
#                     MAX_DELAY * 0.97,
#                     "∞",
#                     ha='center',
#                     va='top',
#                     fontsize=9,
#                     weight='bold'
#                 )

#     # center tick labels under grouped bars
#     ax.set_xticks(x + bar_width * (len(scheduler_order) - 1) / 2)
#     ax.set_xticklabels(flow_order, rotation=25, ha='right')

#     ax.set_ylabel(ylabel)
#     ax.set_xlabel("Flow Type")
#     ax.set_title(f"{ylabel} — {num_flows} Flows, {bandwidth} MHz")
#     ax.set_yticks(np.arange(0, MAX_DELAY, 100))
#     ax.set_ylim(0, MAX_DELAY)

#     ax.legend(
#         loc="upper right",
#         frameon=False
#         # bbox_to_anchor=(1.0, 1.15)
#     )

#     ax.grid(axis='y', linestyle='--', alpha=0.5)

#     # # draw max throughput reference lines (Throughput)
#     # for i, scheduler in enumerate(scheduler_order):
#     #     for j, flow in enumerate(flow_order):
#     #         if flow in max_throughput:
#     #             ax.hlines(
#     #                 y=max_throughput[flow],
#     #                 xmin=x[j] + i * bar_width - bar_width / 2,
#     #                 xmax=x[j] + i * bar_width + bar_width / 2,
#     #                 linestyles='dashed',
#     #                 linewidth=1.5,
#     #                 color='saddlebrown',
#     #                 alpha=0.7
#     #             )
#     #         if flow != 'UE1 Non-GBR' and flow != 'UE3 Non-GBR':
#     #             ax.hlines(
#     #                 y=5,
#     #                 xmin=x[j] + i * bar_width - bar_width / 2,
#     #                 xmax=x[j] + i * bar_width + bar_width / 2,
#     #                 linestyles='dashed',
#     #                 linewidth=1.5,
#     #                 color='black',
#     #                 alpha=0.7
#     #             )

#     # # draw max throughput reference lines (Delay)
#     # for i, scheduler in enumerate(scheduler_order):
#     #     for j, flow in enumerate(flow_order):
#     #         if flow == ("UE1 Non-GBR") or flow == ("UE3 Non-GBR"):
#     #             ax.hlines(
#     #                 y=100,
#     #                 xmin=x[j] + i * bar_width - bar_width / 2,
#     #                 xmax=x[j] + i * bar_width + bar_width / 2,
#     #                 linestyles='dashed',
#     #                 linewidth=1.5,
#     #                 color='black',
#     #                 alpha=0.7
#     #             )
#     #         elif (flow == "UE1 GBR") or (flow == "UE2 GBR"):
#     #             ax.hlines(
#     #                 y=300,
#     #                 xmin=x[j] + i * bar_width - bar_width / 2,
#     #                 xmax=x[j] + i * bar_width + bar_width / 2,
#     #                 linestyles='dashed',
#     #                 linewidth=1.5,
#     #                 color='black',
#     #                 alpha=0.7
#     #             )
#     #         elif (flow == "UE3 DC-GBR"):
#     #             ax.hlines(
#     #                 y=15,
#     #                 xmin=x[j] + i * bar_width - bar_width / 2,
#     #                 xmax=x[j] + i * bar_width + bar_width / 2,
#     #                 linestyles='dashed',
#     #                 linewidth=1.5,
#     #                 color='black',
#     #                 alpha=0.7
#     #             )   
    
#     ax.margins(x=0.02)
#     plt.tight_layout(pad=0.2)

#     plt.savefig(filename, bbox_inches='tight', pad_inches=0.01)
#     plt.show()

# plot_metric(
#     metric='Jitter',
#     ylabel='Jitter [ms]',
#     num_flows=5,
#     bandwidth=5,
#     filename='/home/maximilianrosca/ns-3-dev/jitter_5flows_5mhz.pdf')
# # ------------------- End of Testing plots for paper (throughput, delay, jitter) -------------------------------------





# # ---------------------------------- Plots for learning rate influence on training (PowerPoint) -------------------------------------
# df_lr1e3 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e3_bs32.csv")
# df_lr1e4 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 
# df_lr1e5 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e5_bs32.csv")

# def plot_with_max(ax, df, num_layers, d_model, nhead, color, lr_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     ax.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=2.5,
#         label=f"{lr_label}"
#     )

#     ax.scatter(max_epoch, max_acc, s=55, edgecolor='black', zorder=20)

#     # Determine vertical placement to avoid title overlap
#     ymin, ymax = ax.get_ylim()
#     offset = -18 if max_acc > 0.9 * ymax else 10  # move below if near top

#     ax.annotate(f"{max_acc:.1f}%",
#                 (max_epoch, max_acc),
#                 textcoords="offset points",
#                 xytext=(0, offset),
#                 ha='center',
#                 fontsize=11,
#                 weight='bold')

# plt.rcParams.update({
#     "font.size": 14,
#     "font.family": "sans-serif",
#     "axes.titlesize": 15,
#     "axes.labelsize": 11,
#     "legend.fontsize": 12,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11
# })

# fig, axes = plt.subplots(2, 2, figsize=(9.4, 5), sharex=True)
# axes = axes.flatten()

# # ----------------------
# # Subplot 1 — Model 1 - d_model=16, num_layers=1, num_heads=4, parameters: 79.3k
# # ----------------------
# plot_with_max(axes[0], df_lr1e3, num_layers=1, d_model=16, nhead=4, color='dodgerblue', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-3')
# plot_with_max(axes[0], df_lr1e4, num_layers=1, d_model=16, nhead=4, color='orange', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-4')
# plot_with_max(axes[0], df_lr1e5, num_layers=1, d_model=16, nhead=4, color='limegreen', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-5')
# axes[0].set_ylabel("Validation Accuracy [%]")
# axes[0].set_title("Model 1 (79k params)")
# axes[0].grid(True, linestyle='--', alpha=0.3)
# axes[0].legend(loc='lower right')

# # ------------------------
# # Subplot 2 — Model 2 - d_model=16, num_layers=5, num_heads=4, parameters: 354.3k"
# # ------------------------
# plot_with_max(axes[1], df_lr1e3, num_layers=5, d_model=16, nhead=4, color='dodgerblue', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-3')
# plot_with_max(axes[1], df_lr1e4, num_layers=5, d_model=16, nhead=4, color='orange', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-4')
# plot_with_max(axes[1], df_lr1e5, num_layers=5, d_model=16, nhead=4, color='limegreen', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-5')
# axes[1].set_title("Model 2 (354k params)")
# axes[1].grid(True, linestyle='--', alpha=0.3)
# axes[1].legend(loc='lower right')

# # ------------------------
# # Subplot 3 — Model 3 - d_model=64, num_layers=3, num_heads=4, parameters: 916.2k
# # ------------------------
# plot_with_max(axes[2], df_lr1e3, num_layers=3, d_model=64, nhead=4, color='dodgerblue', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-3')
# plot_with_max(axes[2], df_lr1e4, num_layers=3, d_model=64, nhead=4, color='orange', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-4')
# plot_with_max(axes[2], df_lr1e5, num_layers=3, d_model=64, nhead=4, color='limegreen', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-5')
# axes[2].set_ylabel("Validation Accuracy [%]")
# axes[2].set_title("Model 3 (916k params)")
# axes[2].set_xlabel("Epochs")
# axes[2].grid(True, linestyle='--', alpha=0.3)
# axes[2].legend(loc='lower right')

# # ------------------------
# # Subplot 4 — Model 4 - d_model=128, num_layers=6, num_heads=4, parameters: 3.8M
# # ------------------------
# plot_with_max(axes[3], df_lr1e3, num_layers=6, d_model=128, nhead=4, color='dodgerblue', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-3')
# plot_with_max(axes[3], df_lr1e4, num_layers=6, d_model=128, nhead=4, color='orange', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-4')
# plot_with_max(axes[3], df_lr1e5, num_layers=6, d_model=128, nhead=4, color='limegreen', lr_label='\N{GREEK SMALL LETTER ETA}: 1e-5')
# axes[3].set_title("Model 4 (3.8M params)")
# axes[3].set_xlabel("Epochs")
# axes[3].grid(True, linestyle='--', alpha=0.3)
# axes[3].legend(loc='lower right')

# # Final layout adjustments
# plt.tight_layout(pad=0.3)
# plt.savefig("learning_rate_influence_powerpoint.png",
#             dpi=300,
#             bbox_inches='tight')
# plt.show()
# # ------------------------------- End Plots for learning rate influence on training (PowerPoint)----------------------------------------------





# # ---------------------------------- Plots for number of attention heads influence on training (PowerPoint) -------------------------------------
# df_bs32 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 

# def plot_with_max(ax, df, num_layers, d_model, nhead, lr, color, num_heads_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead) &
#         (df['lr'] == lr)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     ax.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=2.5,
#         label=f"{num_heads_label}"
#     )

#     ax.scatter(max_epoch, max_acc, s=55, edgecolor='black', zorder=20)

#     # Determine vertical placement to avoid title overlap
#     ymin, ymax = ax.get_ylim()
#     offset = -18 if max_acc > 0.9 * ymax else 10  # move below if near top

#     ax.annotate(f"{max_acc:.1f}%",
#                 (max_epoch, max_acc),
#                 textcoords="offset points",
#                 xytext=(0, offset),
#                 ha='center',
#                 fontsize=11,
#                 weight='bold')
   
# plt.rcParams.update({
#     "font.size": 14,
#     "font.family": "sans-serif",
#     "axes.titlesize": 15,
#     "axes.labelsize": 11,
#     "legend.fontsize": 12,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11
# })

# fig, axes = plt.subplots(2, 2, figsize=(9.4, 5), sharex=True)
# axes = axes.flatten()

# # ----------------------
# # Subplot 1 — Model 1 (small) - d_model=16, num_layers=1, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=1, lr=1e-4, color='dodgerblue', num_heads_label='Attention heads: 1')
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=2, lr=1e-4, color='orange', num_heads_label='Attention heads: 2')
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=4, lr=1e-4, color='limegreen', num_heads_label='Attention heads: 4')
# plot_with_max(axes[0], df_bs32, num_layers=1, d_model=16, nhead=8, lr=1e-4, color='red', num_heads_label='Attention heads: 8')

# axes[0].set_ylabel("Validation Accuracy [%]")
# axes[0].set_title("Model 1 (79k params)")
# axes[0].set_yticks(np.arange(68, 80, 2))
# axes[0].grid(True, linestyle='--', alpha=0.3)
# axes[0].legend(loc= "lower right")

# # ----------------------
# # Subplot 2 — Model 2 (small-to-medium) - d_model=16, num_layers=5, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[1], df_bs32, num_layers=5, d_model=16, nhead=1, lr=1e-4, color='dodgerblue', num_heads_label='Attention heads: 1')
# plot_with_max(axes[1], df_bs32, num_layers=5, d_model=16, nhead=2, lr=1e-4, color='orange', num_heads_label='Attention heads: 2')
# plot_with_max(axes[1], df_bs32, num_layers=5, d_model=16, nhead=4, lr=1e-4, color='limegreen', num_heads_label='Attention heads: 4')
# plot_with_max(axes[1], df_bs32, num_layers=5, d_model=16, nhead=8, lr=1e-4, color='red', num_heads_label='Attention heads: 8')
# axes[1].set_title("Model 2 (354k params)")
# axes[1].set_yticks(np.arange(68, 80, 2))
# axes[1].grid(True, linestyle='--', alpha=0.3)
# axes[1].legend(loc= "lower right")

# # ----------------------
# # Subplot 3 — Model 3 (medium-to-large) - d_model=64, num_layers=3, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[2], df_bs32, num_layers=3, d_model=64, nhead=1, lr=1e-4, color='dodgerblue', num_heads_label='Attention heads: 1')
# plot_with_max(axes[2], df_bs32, num_layers=3, d_model=64, nhead=2, lr=1e-4, color='orange', num_heads_label='Attention heads: 2')
# plot_with_max(axes[2], df_bs32, num_layers=3, d_model=64, nhead=4, lr=1e-4, color='limegreen', num_heads_label='Attention heads: 4')
# plot_with_max(axes[2], df_bs32, num_layers=3, d_model=64, nhead=8, lr=1e-4, color='red', num_heads_label='Attention heads: 8')
# axes[2].set_ylabel("Validation Accuracy [%]")
# axes[2].set_title("Model 3 (916k params)")
# axes[2].set_yticks(np.arange(68, 80, 2))
# axes[2].grid(True, linestyle='--', alpha=0.3)
# axes[2].legend(loc= "lower right")
# axes[2].set_xlabel("Epochs")

# # ----------------------
# # Subplot 4 — Model 3 (large) - d_model=128, num_heads=6, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=1, lr=1e-4, color='dodgerblue', num_heads_label='Attention heads: 1')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=2, lr=1e-4, color='orange', num_heads_label='Attention heads: 2')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=4, lr=1e-4, color='limegreen', num_heads_label='Attention heads: 4')
# plot_with_max(axes[3], df_bs32, num_layers=6, d_model=128, nhead=8, lr=1e-4, color='red', num_heads_label='Attention heads: 8')
# axes[3].set_title("Model 4 (3.8M params)")
# axes[3].set_yticks(np.arange(68, 80, 2))
# axes[3].grid(True, linestyle='--', alpha=0.3)
# axes[3].legend(loc= "lower right")
# axes[3].set_xlabel("Epochs")

# # Final layout adjustments
# plt.tight_layout(pad=0.3)
# plt.savefig("attention_heads_influence_powerpoint.png",
#             dpi=300,
#             bbox_inches='tight')
# plt.show()
# # ---------------------------------- End of Plots for number of attention heads influence on training (PowerPoint)-------------------------------------





# # ---------------------------------- Plots for number of encoder layers influence on training (PowerPoint)-------------------------------------
# df_bs32 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 

# def plot_with_max(ax, df, num_layers, d_model, nhead, lr, color, num_layers_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead) &
#         (df['lr'] == lr)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     ax.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=3,
#         alpha=0.9,
#         label=f"{num_layers_label} (max {max_acc:.2f}% @ {int(max_epoch)})"
#     )

#     ax.scatter(max_epoch, max_acc, s=80, edgecolor='black', color=color, zorder=20)

# plt.rcParams.update({
#     "font.size": 18,
#     "font.family": "sans-serif",
#     "axes.titlesize": 20,
#     "axes.labelsize": 18,
#     "legend.fontsize": 14,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14
# })

# fig, axes = plt.subplots(figsize=(9.4, 5))

# # ----------------------
# # Subplot 1 — Model 1 (small) - d_model=16, num_heads=4, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes, df_bs32, num_layers=1, d_model=16, nhead=4, lr=1e-4, color='dodgerblue', num_layers_label='Encoder layers: 1')
# plot_with_max(axes, df_bs32, num_layers=2, d_model=16, nhead=4, lr=1e-4, color='orange', num_layers_label='Encoder layers: 2')
# plot_with_max(axes, df_bs32, num_layers=3, d_model=16, nhead=4, lr=1e-4, color='limegreen', num_layers_label='Encoder layers: 3')
# plot_with_max(axes, df_bs32, num_layers=4, d_model=16, nhead=4, lr=1e-4, color='red', num_layers_label='Encoder layers: 4')
# plot_with_max(axes, df_bs32, num_layers=5, d_model=16, nhead=4, lr=1e-4, color='purple', num_layers_label='Encoder layers: 5')
# plot_with_max(axes, df_bs32, num_layers=6, d_model=16, nhead=4, lr=1e-4, color='brown', num_layers_label='Encoder layers: 6')
# axes.set_ylabel("Validation Accuracy [%]")
# axes.set_xlabel("Epochs")
# axes.set_yticks(np.arange(68, 80, 2))
# axes.set_xticks(np.arange(0, 301, 50))
# axes.grid(True, linestyle='--', alpha=0.35)
# axes.legend(loc= "lower right")

# # Final layout adjustments
# plt.tight_layout(pad=0.3)
# plt.savefig("encoder_layers_influence_powerpoint.png",
#             dpi=300,
#             bbox_inches='tight')
# plt.show()
# # ---------------------------------- End of Plots for number of encoder layers influence on training (PowerPoint)-------------------------------------





# # ---------------------------------- Plots for model dimension influence on training (PowerPoint)-------------------------------------
# df_bs32 = pd.read_csv("/home/maximilianrosca/Masterarbeit/training_dataset6/training_validation_log_file_combined_lr1e4_bs32.csv") 

# def plot_with_max(ax, df, num_layers, d_model, nhead, lr, color, d_model_label):
    
#     filtered = df[
#         (df['num_layers'] == num_layers) &
#         (df['d_model'] == d_model) &
#         (df['nhead'] == nhead) &
#         (df['lr'] == lr)
#     ]
    
#     if filtered.empty:
#         return
    
#     # Find maximum validation accuracy
#     idx_max = filtered['val_acc_symbol'].idxmax()
#     max_epoch = filtered.loc[idx_max, 'epoch']
#     max_acc = filtered.loc[idx_max, 'val_acc_symbol']
    
#     # Plot curve with max info in label
#     ax.plot(
#         filtered['epoch'],
#         filtered['val_acc_symbol'],
#         color=color,
#         linewidth=3,
#         alpha=0.9,
#         label=f"{d_model_label}"
#     )

#     ax.scatter(max_epoch, max_acc, s=80, edgecolor='black', color=color, zorder=20)

#     # smart offset to avoid top collision
#     ymin, ymax = ax.get_ylim()
#     offset = -20 if max_acc > 0.9 * ymax else 12

#     ax.annotate(f"{max_acc:.2f}%",
#                 (max_epoch, max_acc),
#                 textcoords="offset points",
#                 xytext=(0, offset),
#                 ha='center',
#                 fontsize=13,
#                 weight='bold')
    
# plt.rcParams.update({
#     "font.size": 18,
#     "font.family": "sans-serif",
#     "axes.titlesize": 20,
#     "axes.labelsize": 18,
#     "legend.fontsize": 14,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14
# })

# fig, axes = plt.subplots(figsize=(9.4, 5))

# # ----------------------
# # Subplot 1 — Model 1 (small) - num_layers=1, nheads=4, l_r=1e-4, batch_size=32, epochs=300, dataset=346.866 samples
# # ----------------------
# plot_with_max(axes, df_bs32, num_layers=1, d_model=16, nhead=4, lr=1e-4, color='dodgerblue', d_model_label='Model dimension: 16')
# plot_with_max(axes, df_bs32, num_layers=1, d_model=32, nhead=4, lr=1e-4, color='orange', d_model_label='Model dimension: 32')
# plot_with_max(axes, df_bs32, num_layers=1, d_model=64, nhead=4, lr=1e-4, color='limegreen', d_model_label='Model dimension: 64')
# plot_with_max(axes, df_bs32, num_layers=1, d_model=128, nhead=4, lr=1e-4, color='red', d_model_label='Model dimension: 128')

# axes.set_ylabel("Validation Accuracy [%]")
# axes.set_xlabel("Epochs")
# axes.set_yticks(np.arange(68, 80, 2))
# axes.set_xticks(np.arange(0, 301, 50))
# axes.grid(True, linestyle='--', alpha=0.35)
# axes.legend(loc= "lower right")

# # Final layout adjustments
# plt.tight_layout(pad=0.3)
# plt.savefig("model_dimension_influence_powerpoint.png",
#             dpi=300,
#             bbox_inches='tight')
# plt.show()
# # ---------------------------------- End of Plots for model dimension influence on training (PowerPoint)-------------------------------------





# # --------------------------- Testing plots for PowerPoint (throughput, delay, jitter) -------------------------------------------
# df = pd.read_csv('/home/maximilianrosca/ns-3-dev/simulation_results_4flows_10mhz.csv')
# df.columns = df.columns.str.strip()
# df['FlowType'] = (
#     df['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )

# df_baseline = pd.read_csv('/home/maximilianrosca/ns-3-dev/simulation_results_4flows_10mhz_mlp.csv')
# df_baseline.columns = df_baseline.columns.str.strip()
# df_baseline['FlowType'] = (
#     df_baseline['FlowType']
#       .str.strip()
#       .str.replace('\u00a0', ' ', regex=False)  # non-breaking space
#       .str.replace('–', '-', regex=False)       # long dash
# )
# df_baseline['scheduler'] = df_baseline['scheduler'].str.replace('ns3::NrMacSchedulerTdmaQos', 'MLP', regex=False)

# df = pd.concat([df, df_baseline], ignore_index=True)

# def scheduler_label(row):
#     if row['scheduler'].endswith('RR'):
#         return 'RR'
#     elif row['scheduler'].endswith('PF'):
#         return 'PF'
#     elif row['scheduler'].endswith('Qos'):
#         if row['ML'] == 0:
#             return 'QoS'
#         else:
#             return "TF QoS"
#     elif row['scheduler'] == 'MLP':
#         return 'MLP'

# df['SchedulerLabel'] = df.apply(scheduler_label, axis=1)

# flow_order = [
#     'UE1 Non-GBR',
#     'UE1 GBR',
#     'UE2 GBR',
#     # 'UE3 Non-GBR',
#     'UE3 DC-GBR'
# ]

# scheduler_order = ['QoS', 'TF QoS', 'MLP']

# colors = {
#     'QoS': 'tab:green',
#     'TF QoS': 'tab:red',
#     'MLP': 'tab:purple'
# }

# plt.rcParams.update({
#     "font.family": "sans-serif",
#     "font.size": 11,
#     "axes.titlesize": 11,
#     "axes.labelsize": 10,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "legend.fontsize": 8,
# })

# max_throughput = {
#     'UE1 Non-GBR': 5,
#     'UE1 GBR': 10,
#     'UE2 GBR': 10,
#     # 'UE3 Non-GBR': 5,
#     'UE3 DC-GBR': 15
# }

# def plot_metric(metric, ylabel, num_flows, bandwidth, filename):

#     fig, ax = plt.subplots(figsize=(4.8, 3.2))

#     bar_width = 0.25
#     # x = np.arange(len(flow_order))
#     x = np.array([0, 1, 2.3, 3.3])
#     # x = np.array([0, 1, 2.4, 3.6, 4.6])


#     for i, scheduler in enumerate(scheduler_order):
#         for j, flow in enumerate(flow_order):

#             row = df[
#                 (df['FlowType'] == flow) &
#                 (df['SchedulerLabel'] == scheduler)
#             ]

#             MAX_DELAY = 3000

#             # if row[metric].values[0] == 0:
#             #     value = MAX_DELAY
#             #     is_missing = True
#             # else:
#             #     value = row[metric].values[0]
#             #     is_missing = False
#             value = row[metric].values[0]
#             is_missing = False
#             ax.bar(
#                 x[j] + i * bar_width,
#                 value,
#                 width=bar_width,
#                 color=colors[scheduler],
#                 edgecolor='black' if is_missing else None,
#                 hatch='//' if is_missing else None,
#                 # label=scheduler if ((j == 0) and not is_missing) or (j == 3 and scheduler == 'MLP') else None
#                 label = scheduler if (j == 0) else None
#             )

#             if is_missing:
#                 ax.text(
#                     x[j] + i * bar_width,
#                     MAX_DELAY * 0.97,
#                     "∞",
#                     ha='center',
#                     va='top',
#                     fontsize=9,
#                     weight='bold'
#                 )

#     # center tick labels under grouped bars
#     ax.set_xticks(x + bar_width * (len(scheduler_order) - 1) / 2)
#     ax.set_xticklabels(flow_order, rotation=15, ha='right')
#     ax.set_title(f"{bandwidth} MHz Bandwidth Scenario", pad=4)
#     # ax.set_yticks(np.arange(0, MAX_DELAY, 500))
#     # ax.set_ylim(0, MAX_DELAY)

#     ax.set_ylabel(ylabel)
#     # ax.set_xlabel("Flow Type")

#     ax.legend(
#         loc="upper left",
#         frameon=False
#         # bbox_to_anchor=(1.0, 1.15)
#     )

#     ax.grid(axis='y', linestyle='--', alpha=0.5)

#     # draw max throughput reference lines (Throughput)
#     for i, scheduler in enumerate(scheduler_order):
#         for j, flow in enumerate(flow_order):
#             if flow in max_throughput:
#                 ax.hlines(
#                     y=max_throughput[flow],
#                     xmin=x[j] + i * bar_width - bar_width / 2,
#                     xmax=x[j] + i * bar_width + bar_width / 2,
#                     linestyles='dashed',
#                     linewidth=1.5,
#                     color='saddlebrown',
#                     alpha=0.7
#                 )
#             if flow != 'UE1 Non-GBR':
#                 ax.hlines(
#                     y=5,
#                     xmin=x[j] + i * bar_width - bar_width / 2,
#                     xmax=x[j] + i * bar_width + bar_width / 2,
#                     linestyles='dashed',
#                     linewidth=1.5,
#                     color='black',
#                     alpha=0.7
#                 )

#     # # draw max throughput reference lines (Delay)
#     # for i, scheduler in enumerate(scheduler_order):
#     #     for j, flow in enumerate(flow_order):
#     #         if flow == "UE1 Non-GBR":
#     #             ax.hlines(
#     #                 y=100,
#     #                 xmin=x[j] + i * bar_width - bar_width / 2,
#     #                 xmax=x[j] + i * bar_width + bar_width / 2,
#     #                 linestyles='dashed',
#     #                 linewidth=1.5,
#     #                 color='black',
#     #                 alpha=0.7
#     #             )
#     #         elif (flow == "UE1 GBR") or (flow == "UE2 GBR"):
#     #             ax.hlines(
#     #                 y=300,
#     #                 xmin=x[j] + i * bar_width - bar_width / 2,
#     #                 xmax=x[j] + i * bar_width + bar_width / 2,
#     #                 linestyles='dashed',
#     #                 linewidth=1.5,
#     #                 color='black',
#     #                 alpha=0.7
#     #             )
#     #         elif (flow == "UE3 DC-GBR"):
#     #             ax.hlines(
#     #                 y=15,
#     #                 xmin=x[j] + i * bar_width - bar_width / 2,
#     #                 xmax=x[j] + i * bar_width + bar_width / 2,
#     #                 linestyles='dashed',
#     #                 linewidth=1.5,
#     #                 color='black',
#     #                 alpha=0.7
#     #             )   
         
#     ax.margins(x=0.02)
#     plt.tight_layout(pad=0.15)
#     plt.savefig(filename,
#                 dpi=300,
#                 bbox_inches='tight',
#                 pad_inches=0.02)
#     plt.show()

# plot_metric(
#     metric='Throughput',
#     ylabel='Throughput [Mbps]',
#     num_flows=4,
#     bandwidth=10,
#     filename='/home/maximilianrosca/Masterarbeit/Final-Presentation/Figures/Throughput-10mhz-powerpoint.png')
# # ---------------------------End of Testing plots for PowerPoint (throughput, delay, jitter) -------------------------------------------