#!/bin/bash

PREFIX="$1"      # e.g., ring
USER_HOST="your_username@grill.win.tue.nl:/your_path/eval_fixedinit"
MAX_JOBS=6       # maximum concurrent SCP transfers

samples=(100 200 300 400 600 700 800 900)

job_count=0

for n in "${samples[@]}"; do
    folder="${PREFIX}_${n}"
    if [ -d "$folder" ]; then
        echo "Transferring $folder..."
        scp -r "$folder" "$USER_HOST/$folder" &  # run in background
        ((job_count++))
        
        # wait if MAX_JOBS are running
        if (( job_count >= MAX_JOBS )); then
            wait -n  # waits for any background job to finish
            ((job_count--))
        fi
    else
        echo "Folder $folder does not exist yet. Skipping."
    fi
done

# wait for all remaining jobs
wait
echo "All SCP transfers done."
