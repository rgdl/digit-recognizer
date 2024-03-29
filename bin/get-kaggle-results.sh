#!/usr/bin/env bash

kernel=$(jq '.id' gen/kernel-metadata.json | awk -F'"' '{print $2}')
kernel_status=$(kaggle kernels status $kernel | awk -F'"' '{print $2}')

if [ $kernel_status = "queued" ]; then
    echo "Still queued..."
elif [ $kernel_status = "running" ]; then
    echo "Still running..."
else
    echo "Status is now $kernel_status"
    kaggle kernels output -p data/output/kaggle_logs $kernel
    python bin/read_log.py data/output/kaggle_logs/*.log
fi

