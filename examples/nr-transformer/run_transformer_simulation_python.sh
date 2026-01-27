#!/bin/bash

# scenarios=("RMa" "UMa")
randomStream=1

for i in $(seq 1 10); do
    # for scenario in "${scenarios[@]}"; do
      # echo "Running with randomStream=$randomStream, scenario=$scenario"
    echo "Running with randomStream=$randomStream"
    python nr_transformer_model_inference.py \
      --randomStream=$randomStream \
        # --scenario=$scenario
    sleep 1
    ((randomStream++))
    # done
done