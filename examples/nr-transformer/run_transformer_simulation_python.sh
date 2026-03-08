#!/bin/bash

scenarios=("UMa")
bandwidths=(50000000)
randomStream=4


for scenario in "${scenarios[@]}"; do
  for bandwidth in "${bandwidths[@]}"; do
    echo "Running with randomStream=$randomStream, scenario=$scenario, bandwidth=$bandwidth"
    python nr_baseline_model_inference.py \
    --randomStream=$randomStream \
    --scenario=$scenario \
    --enableTransformer=True \
    --bandwidth=$bandwidth
    sleep 1
    ((randomStream++))
  done
done
