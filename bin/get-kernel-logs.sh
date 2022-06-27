#!/usr/bin/env bash -e

working_dir=/tmp/kaggle-log
kernel_id=$1
logfile=$working_dir/$(echo $1 | awk -F'/' '{print $2}').log

function cleanup {
    rm -r $working_dir
}

trap cleanup EXIT

kaggle kernels output $kernel_id --path $working_dir

# Output results
cat $logfile \
    | jq '.[] | select(.stream_name | contains("stdout")) .data' \
    | sed "s/^\"//g" | sed "s/\\\n\"$//g"

cat << EOF >&2
$(
cat $logfile \
    | jq '.[] | select(.stream_name | contains("stderr")) .data' \
    | sed "s/^\"//g" | sed "s/\\\n\"$//g"
)
EOF
