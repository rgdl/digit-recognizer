#!/usr/bin/env bash

kaggle kernels init -p gen

jq '. + {
    id: "pumpkin/digit-recognizer",
    title: "Digit Recognizer",
    code_file: "main.py",
    language: "python",
    kernel_type: "script",
    enable_gpu: "true",
    enable_internet: "false",
    competition_sources: ["digit-recognizer"] 
}' gen/kernel-metadata.json > tmp.json
mv tmp.json gen/kernel-metadata.json
