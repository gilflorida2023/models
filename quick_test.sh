#!/bin/bash

# identify tools models
rm -f toolsmodels.txt
for i in $(ollama list | awk 'NR>1 {print $1}'); do
    if ollama show "$i" | grep -iq "tools"; then
        echo "$i" >> toolsmodel.txt
    fi
done

for i in $(cat toolsmodel.txt); do
    name="${i//:/_}"          # replace colons with underscores
    name="${name//\//_}"      # replace slashes with underscores
    outfile="${name}.txt"     # add .txt extension
    thinking_outfile="thinking.${name}.txt"

    echo "$i" > "$outfile"
    echo "=====  s h o w  ==========================================================" >> "$outfile"
    ollama show "$i" | tee -a "$outfile"

    echo "=====  p r o m p t . i n f o  ==================================" >> "$outfile"
    cat prompt1.info >> "$outfile"

    echo "=====  r u n  = not thinking ====================================" >> "$outfile"

    # Warmup run (not timed) - loads model into memory
    echo "=====  w a r m u p  ==============================================" >> "$outfile"
    ollama run "$i" --think=false < prompt1.info >/dev/null 2>&1

    # Stop after warmup
    ollama stop "$i" 2>/dev/null
    sleep 2

    # === Measure elapsed time ===
    start=$(date +%s.%N)                    # start time with nanoseconds

    ollama run "$i" --verbose --think=false < prompt1.info 2>&1 | ansifilter | tee -a "$outfile"

    end=$(date +%s.%N)                      # end time
    elapsed=$(echo "$end - $start" | bc -l) # calculate difference

    # Human readable time
    minutes=$(echo "$elapsed / 60" | bc)
    seconds=$(echo "$elapsed % 60" | bc)

    if (( minutes > 0 )); then
        printf "\n=== Elapsed time: %d minutes %.2f seconds ===\n" "$minutes" "$seconds" | tee -a "$outfile"
    else
        printf "\n=== Elapsed time: %.2f seconds ===\n" "$elapsed" | tee -a "$outfile"
    fi


    # Stop ALL running models, wait until none are left
    while true; do
        ollama ps 2>/dev/null | tail -n +2 | while read -r model _; do
            [[ -n "$model" ]] && ollama stop "$model" 2>/dev/null
        done
        sleep 2
        if ! ollama ps 2>/dev/null | tail -n +2 | grep -q .; then
            break
        fi
    done

    if ollama show "$i" | grep -iq "thinking"; then

        echo "=====  r u n  = thinking ========================================" >> "$thinking_outfile"

        # Warmup run (not timed) - loads model into memory
        echo "=====  w a r m u p  ==============================================" >> "$thinking_outfile"
        ollama run "$i" --think=true < prompt1.info >/dev/null 2>&1

        # Stop ALL running models before timed thinking run
        while true; do
            ollama ps 2>/dev/null | tail -n +2 | while read -r model _; do
                [[ -n "$model" ]] && ollama stop "$model" 2>/dev/null
            done
            sleep 2
            if ! ollama ps 2>/dev/null | tail -n +2 | grep -q .; then
                break
            fi
        done

        # === Measure elapsed time ===
        start=$(date +%s.%N)                    # start time with nanoseconds

        ollama run "$i" --verbose --think=true < prompt1.info 2>&1 | ansifilter | tee -a "$thinking_outfile"

        end=$(date +%s.%N)                      # end time
        elapsed=$(echo "$end - $start" | bc -l) # calculate difference

        # Human readable time
        minutes=$(echo "$elapsed / 60" | bc)
        seconds=$(echo "$elapsed % 60" | bc)

        if (( minutes > 0 )); then
            printf "\n=== Elapsed time: %d minutes %.2f seconds ===\n" "$minutes" "$seconds" | tee -a "$thinking_outfile"
        else
            printf "\n=== Elapsed time: %.2f seconds ===\n" "$elapsed" | tee -a "$thinking_outfile"
        fi

    else
        echo "=====  r u n  = thinking ===== (skipped - model does not support thinking) =====" | tee -a "$thinking_outfile"
    fi

done