#!/usr/bin/env bash

kernel_id=$1
max_status_checks=10
wait_interal=2

for check in $(seq $max_status_checks)
do
    >&2 echo waiting...
    sleep $wait_interal
    [[ $(kaggle kernels status $kernel_id) =~ \"complete\" ]] && break
done

if [[ $check = $max_status_checks ]]
then
    >&2 echo Timeout!
    exit 1
fi
