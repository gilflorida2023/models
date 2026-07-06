#!/bin/bash

for i in $(ollama list | awk 'NR>1 {print $1}'); do
    name="${i//:/_}.txt"

    echo "$i" > "$name"
    echo "=====  s h o w  ==========================================================" >> "$name"
    ollama show "$i" | tee -a "$name"

    echo "=====  p r o m p t . i n f o  ==================================" >> "$name"
    cat prompt.info >> "$name"

    echo "=====  run  = not thinking ====================================" >> "$name"

    # === Measure elapsed time ===
    start=$(date +%s.%N)                    # start time with nanoseconds

    ollama run "$i" --verbose --think=false < prompt.info 2>&1 | ansifilter | tee -a "$name"

    end=$(date +%s.%N)                      # end time
    elapsed=$(echo "$end - $start" | bc -l) # calculate difference

    # Human readable time
    minutes=$(echo "$elapsed / 60" | bc)
    seconds=$(echo "$elapsed % 60" | bc)

    if (( minutes > 0 )); then
        printf "\n=== Elapsed time: %d minutes %.2f seconds ===\n" "$minutes" "$seconds" | tee -a "$name"
    else
        printf "\n=== Elapsed time: %.2f seconds ===\n" "$elapsed" | tee -a "$name"
    fi


    ollama stop  "$i"
    sleep 2
    ollama stop "$i"
    sleep 2
    ollama stop "$i"
    sleep 2

    echo "=====  run  = thinking ========================================" >> "thinking.$name"

    # === Measure elapsed time ===
    start=$(date +%s.%N)                    # start time with nanoseconds

    ollama run "$i" --verbose --think=true < prompt.info 2>&1 | ansifilter | tee -a "thinking.${name}"

    end=$(date +%s.%N)                      # end time
    elapsed=$(echo "$end - $start" | bc -l) # calculate difference

    # Human readable time
    minutes=$(echo "$elapsed / 60" | bc)
    seconds=$(echo "$elapsed % 60" | bc)

    if (( minutes > 0 )); then
        printf "\n=== Elapsed time: %d minutes %.2f seconds ===\n" "$minutes" "$seconds" | tee -a "thnking.$name"
    else
        printf "\n=== Elapsed time: %.2f seconds ===\n" "$elapsed" | tee -a "thinking.$name"
    fi



done
