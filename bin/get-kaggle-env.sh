#!/usr/bin/env bash -e

# Run `pip freeze` in a kaggle kernel to get the latest kaggle environment

upload_dir=/tmp/kaggle
script_dir=$(dirname "$0")
kernel_id=pumpkin/get-kaggle-env
max_status_checks=10

function cleanup {
    rm  -r $upload_dir
}

trap cleanup EXIT

# Push code to kernel

mkdir $upload_dir

cat <<EOF > $upload_dir/kernel-metadata.json
{
  "id": "$kernel_id",
  "id_no": 12345,
  "title": "Get Kaggle Env",
  "code_file": "run-pip-freeze.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "false",
  "enable_gpu": "false",
  "enable_internet": "false",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
EOF

cp $script_dir/run-pip-freeze.py $upload_dir
kaggle kernels push --path $upload_dir

$script_dir/await_kernel_completion.sh $kernel_id
$script_dir/get-kernel-logs.sh $kernel_id
