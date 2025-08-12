import pandas as pd     
import matplotlib.pyplot as plt

file = pd.read_csv("/home/maximilianrosca/ns-3-dev/simulation_results.csv")

file['scheduler'] = file['scheduler'].str.replace('ns3::NrMacScheduler', '')
file['FlowId'] = file['FlowId'].astype(str).str.replace('1', 'NGBR_LOW_LAT_EMBB')
file['FlowId'] = file['FlowId'].astype(str).str.replace('2', 'GBR_CONV_VOICE')

grouped_scenario_scheduler = file.groupby(['scenario', 'scheduler'])
summary_scenario_scheduler = grouped_scenario_scheduler[['Throughput', 'Delay', 'Jitter']].mean().reset_index()

pivot = summary_scenario_scheduler.pivot(index='scenario', columns='scheduler', values='Delay')
pivot.plot(kind='bar', figsize=(8, 6), width=0.75)

plt.ylabel("Average Delay [ms]")
plt.xlabel("Scenario")
plt.title("Delay by Scheduler Type and Scenario")
plt.xticks(rotation=0)
plt.legend(title="Scheduler")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("delay_by_scheduler_and_scenario.png")

pivot = summary_scenario_scheduler.pivot(index='scenario', columns='scheduler', values='Throughput')
pivot.plot(kind='bar', figsize=(8, 6), width=0.75)

plt.ylabel("Average Throughput [Mbps]")
plt.xlabel("Scenario")
plt.title("Throughput by Scheduler Type and Scenario")
plt.xticks(rotation=0)
plt.legend(title="Scheduler")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("throughput_by_scheduler_and_scenario.png")



grouped_scheduler_flow = file.groupby(['scheduler', 'FlowId'])
summary_scheduler_flow = grouped_scheduler_flow[['Throughput', 'Delay', 'Jitter']].mean().reset_index()

pivot = summary_scheduler_flow.pivot(index='FlowId', columns='scheduler', values='Delay')
pivot.plot(kind='bar', figsize=(8, 6), width=0.75)

plt.ylabel("Average Delay [ms]")
plt.xlabel("Flow ID")
plt.title("Delay by Scheduler Type and Flow ID")
plt.xticks(rotation=0)
plt.legend(title="Scheduler")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("delay_by_scheduler_and_flow_id.png")

pivot = summary_scheduler_flow.pivot(index='FlowId', columns='scheduler', values='Throughput')
pivot.plot(kind='bar', figsize=(8, 6), width=0.75)

plt.ylabel("Average Throughput [Mbps]")
plt.xlabel("Flow ID")
plt.title("Throughput by Scheduler Type and Flow ID")
plt.xticks(rotation=0)
plt.legend(title="Scheduler")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("throughput_by_scheduler_and_flow_id.png")



rma_values =  file[file["scenario"] == "RMa"]
rma_scheduler_flow = file.groupby(['scheduler', 'FlowId'])
rma_scheduler_flow = rma_scheduler_flow[['Throughput', 'Delay', 'Jitter']].mean().reset_index()
pivot = rma_scheduler_flow.pivot(index='FlowId', columns='scheduler', values='Delay')
pivot.plot(kind='bar', figsize=(8, 6), width=0.75)

plt.ylabel("Average Delay [ms]")
plt.xlabel("Flow ID")
plt.title("Delay by Scheduler Type and Flow ID in Rma Scenario")
plt.xticks(rotation=0)
plt.legend(title="Scheduler")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("rma_delay_by_scheduler_and_flow_id.png")

pivot = rma_scheduler_flow.pivot(index='FlowId', columns='scheduler', values='Throughput')
pivot.plot(kind='bar', figsize=(8, 6), width=0.75)

plt.ylabel("Average Throughput [Mbps]")
plt.xlabel("Flow ID")
plt.title("Throughput by Scheduler Type and Flow ID in Rma Scenario")
plt.xticks(rotation=0)
plt.legend(title="Scheduler")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("rma_throughput_by_scheduler_and_flow_id.png")



uma_values =  file[file["scenario"] == "UMa"]
uma_scheduler_flow = file.groupby(['scheduler', 'FlowId'])
uma_scheduler_flow = uma_scheduler_flow[['Throughput', 'Delay', 'Jitter']].mean().reset_index()

pivot = uma_scheduler_flow.pivot(index='FlowId', columns='scheduler', values='Delay')
pivot.plot(kind='bar', figsize=(8, 6), width=0.75)

plt.ylabel("Average Delay [ms]")
plt.xlabel("Flow ID")
plt.title("Delay by Scheduler Type and Flow ID in UMa Scenario")
plt.xticks(rotation=0)
plt.legend(title="Scheduler")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("uma_delay_by_scheduler_and_flow_id.png")

pivot = uma_scheduler_flow.pivot(index='FlowId', columns='scheduler', values='Throughput')
pivot.plot(kind='bar', figsize=(8, 6), width=0.75)

plt.ylabel("Average Throughput [Mbps]")
plt.xlabel("Flow ID")
plt.title("Throughput by Scheduler Type and Flow ID in UMa Scenario")
plt.xticks(rotation=0)
plt.legend(title="Scheduler")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("uma_throughput_by_scheduler_and_flow_id.png")



umi_values =  file[file["scenario"] == "UMi"]
umi_scheduler_flow = file.groupby(['scheduler', 'FlowId'])
umi_scheduler_flow = umi_scheduler_flow[['Throughput', 'Delay', 'Jitter']].mean().reset_index()

pivot = umi_scheduler_flow.pivot(index='FlowId', columns='scheduler', values='Delay')
pivot.plot(kind='bar', figsize=(8, 6), width=0.75)

plt.ylabel("Average Delay [ms]")
plt.xlabel("Flow ID")
plt.title("Delay by Scheduler Type and Flow ID in UMi Scenario")
plt.xticks(rotation=0)
plt.legend(title="Scheduler")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("umi_delay_by_scheduler_and_flow_id.png")

pivot = umi_scheduler_flow.pivot(index='FlowId', columns='scheduler', values='Throughput')
pivot.plot(kind='bar', figsize=(8, 6), width=0.75)

plt.ylabel("Average Throughput [Mbps]")
plt.xlabel("Flow ID")
plt.title("Throughput by Scheduler Type and Flow ID in UMi Scenario")
plt.xticks(rotation=0)
plt.legend(title="Scheduler")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("umi_throughput_by_scheduler_and_flow_id.png")
