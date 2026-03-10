#!/bin/bash

# Usage: bash TrainSet_curve.sh <beamshape> <learning_rate>
# Example: bash TrainSet_curve_sddal.sh rec 0.0002

BEAMSHAPE="$1"
LR="$2"

if [ -z "$BEAMSHAPE" ] || [ -z "$LR" ]; then
    echo "Usage: $0 <beamshape> <learning_rate>"
    exit 1
fi

# List of num_samples
samples=(200 300 700 1000 400 500 800 900)

idx=0
for n in "${samples[@]}"; do

    # Compute n-100 (only for naming)
    n_minus_100=$((n - 100))

    # First 5 jobs -> gpu0, rest -> gpu1, set -lt as 999 to run all jobs on gpu0, -lt as 0 to run all jobs on gpu1
    if [ "$idx" -lt 4 ]; then
        gpu=0
    else
        gpu=1
    fi
    idx=$((idx + 1))

    # Use n_minus_100 for naming
    VAL_VIS_DIR="${BEAMSHAPE}_${n_minus_100}"
    out_file="${BEAMSHAPE}_${n_minus_100}.txt"
    pth_name="${BEAMSHAPE}_${n_minus_100}.pth.tar"

    echo "Launching training for beamshape=${BEAMSHAPE}, num_samples=${n} (named as ${n_minus_100}) on GPU $gpu"

    nohup python3 train_unet.py \
        --data "../Design_${BEAMSHAPE}" \
        --epochs 15 \
        --batch_size 2 \
        --gpu "$gpu" \
        --lr "$LR" \
        --step_size 2 \
        --seed 123 \
        --pth_name "$pth_name" \
        --val_vis_path "$VAL_VIS_DIR" \
        --num_samples "$n" \
        > "$out_file" 2>&1 &
done

echo "All training jobs launched."
