#!/bin/bash

scenarios=("RMa" "UMa" "UMi")

for randomStream in {1..5}; do
  for numerology in {0..2}; do
    for scenarioType in "${scenarios[@]}"; do
      echo "Running with randomStream=$randomStream, numerology=$numerology, scenario=$scenarioType"
      python3 nr_transformer_use_model.py \
        --randomStream=$randomStream \
        --numerology=$numerology \
        --scenario=$scenarioType

      sleep 1
    done
  done
done
