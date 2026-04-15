#!/bin/bash

# ==========================================
# Usage:
#   bash SDDAL.sh <beamshape> <lr> <initial_size> <init_only?> <start_round> <end_round> <gpu> <scanner_batch_size> <retrain_frequency> <scan_only?> <update_size> <test_data>
#
# Example:
#   regular scheme: bash SDDAL.sh rec 0.0002 100 false 1 580 0 5 1 false 50 Design_rec
#   eval disabled:  bash SDDAL.sh rec 0.0002 100 false 1 580 0 5 1 false 0 ""
#   scan-only scheme: run the following sequence to do 1000initial+1000SDDAL
#                   bash SDDAL.sh rec 0.0002 1000 true 9999 9999 0 9999 9999
#                   bash SDDAL.sh rec 9999 9999 false 1 200 0 5 9999 true 0 ""
#
# Evaluation args (optional, set update_size=0 to disable):
#   update_size : evaluate every N new training samples (default 50)
#   test_data   : directory containing test_set/intensity/npy and test_set/phase/npy
#                 (same dir you pass as --train_data to Trainer, e.g. Design_rec)
# ==========================================

beamshape=${1:-chair}
lr=${2:-0.0002}
init_size=${3:-100}
init_only=${4:-false}
start_round=${5:-1}
end_round=${6:-580}
gpu=${7:-0}
scanner_batch_size=${8:-5}
retrain_freq=${9:-1}
scan_only=${10:-false}
update_size=${11:-50}    # evaluate every N new samples; set to 0 to disable
test_data=${12:-""}      # dir with test_set/intensity/npy and test_set/phase/npy

trainer_batch_size=2  # fixed for Trainer.py

echo "========================================="
echo " Neural Experimental Design (NED) Pipeline"
echo " Beamshape            : ${beamshape}"
echo " Learning rate        : ${lr}"
echo " Rounds               : ${start_round} → ${end_round}"
echo " GPU                  : ${gpu}"
echo " Trainer batch size   : ${trainer_batch_size}"
echo " Scanner batch size   : ${scanner_batch_size}"
echo " Initial set size     : ${init_size}"
echo " Retrain frequency    : ${retrain_freq}"
echo " Scan only?           : ${scan_only}"
echo " Init only?           : ${init_only}"
echo " Eval update size     : ${update_size} (0 = disabled)"
echo " Test data dir        : ${test_data:-'(not set)'}"
echo "========================================="

# --- Handle init_only=true separately ---
if [ "${init_only}" = true ]; then
    echo "------------------------------"
    echo "  init_only=true → Generate initial training set and train once"
    echo "------------------------------"

    echo "  Running Initializer.py..."
    python3 Initializer.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --init_size ${init_size} \
        --vis_path Design_${beamshape}

    echo "  Training model on initial set..."
    python3 Trainer.py \
        --train_data Design_${beamshape} \
        --epochs 15 \
        --batch_size ${trainer_batch_size} \
        --gpu ${gpu} \
        --lr ${lr} \
        --step_size 2 \
        --seed 123 \
        --pth_name Design_${beamshape}/models/QuantUNetT_${beamshape}

    echo "------------------------------"
    echo "  init_only pipeline finished."
    echo "------------------------------"
    exit 0
fi

# --- Regular behavior (init_only=false) ---
# Run Initializer.py only when starting from round 1 AND scan_only=false
if [ "${scan_only}" = true ]; then
    echo "------------------------------"
    echo "  scan_only=true → Skipping Initializer.py"
    echo "------------------------------"

elif [ "${start_round}" -eq 1 ]; then
    echo "------------------------------"
    echo "  Running Initializer.py"
    echo "------------------------------"

    python3 Initializer.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --init_size ${init_size} \
        --vis_path Design_${beamshape}

else
    echo "------------------------------"
    echo "  Skipping Initializer.py (resuming from round ${start_round})"
    echo "------------------------------"
fi

# --- Timing and dataset size tracking ---
# dataset_size accounts for samples already collected before start_round
dataset_size=$(( init_size + (start_round - 1) * scanner_batch_size ))
loop_start_s=$(date +%s)
cumul_scanner_s=0
cumul_trainer_s=0
cumul_eval_s=0

# Loop over rounds
for ((round_sampling=${start_round}; round_sampling<=${end_round}; round_sampling++))
do
    echo "------------------------------"
    echo "  Starting Round ${round_sampling}  (dataset_size=${dataset_size})"
    echo "------------------------------"

    # --- Trainer (with timing) ---
    if [ "${scan_only}" = false ]; then
        if (( (round_sampling - 1) % retrain_freq == 0 )); then
            echo "------------------------------"
            echo "  Re-training model at round ${round_sampling}"
            echo "  (Training happens every ${retrain_freq} scans)"
            echo "------------------------------"

            t0=$(date +%s)
            python3 Trainer.py \
                --train_data Design_${beamshape} \
                --epochs 15 \
                --batch_size ${trainer_batch_size} \
                --gpu ${gpu} \
                --lr ${lr} \
                --step_size 2 \
                --seed 123 \
                --pth_name Design_${beamshape}/models/QuantUNetT_${beamshape}
            t1=$(date +%s)
            cumul_trainer_s=$(( cumul_trainer_s + t1 - t0 ))
        else
            echo "  Skipping training at this round (waiting for next frequency point)"
        fi
    else
        echo "  scan_only=true → training skipped."
    fi

    # --- Scanner (with timing) ---
    t0=$(date +%s)
    python3 Scanner.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --batch_size ${scanner_batch_size} \
        --pth_name QuantUNetT_${beamshape} \
        --round_sampling ${round_sampling} \
        --vis_path Design_${beamshape}
    t1=$(date +%s)
    cumul_scanner_s=$(( cumul_scanner_s + t1 - t0 ))
    dataset_size=$(( dataset_size + scanner_batch_size ))

    # --- Inline evaluation ---
    if [ "${update_size}" -gt 0 ] && [ -n "${test_data}" ] && (( dataset_size % update_size == 0 )); then
        wall_s=$(( $(date +%s) - loop_start_s ))
        checkpoint="Design_${beamshape}/models/QuantUNetT_${beamshape}.pth.tar"
        log_file="Design_${beamshape}/eval_log.csv"
        echo "  [eval] dataset_size=${dataset_size} → running evaluation..."
        t0=$(date +%s)
        python3 evaluate_checkpoint.py \
            --checkpoint      "${checkpoint}" \
            --test_data       "${test_data}" \
            --gpu             "${gpu}" \
            --log_file        "${log_file}" \
            --round           "${round_sampling}" \
            --dataset_size    "${dataset_size}" \
            --wall_clock_s    "${wall_s}" \
            --cumul_scanner_s "${cumul_scanner_s}" \
            --cumul_trainer_s "${cumul_trainer_s}" \
            --cumul_eval_s    "${cumul_eval_s}"
        t1=$(date +%s)
        cumul_eval_s=$(( cumul_eval_s + t1 - t0 ))
    fi

done

# Zernike coefficients statistics
echo "========================================="
echo "   Zernike coefficient statistics in progress…"
echo "========================================="
python3 zernike_statistics.py --beamshape ${beamshape} --init_size ${init_size}

echo "========================================="
echo "   SDDAL (Simulation-Driven Differentiable Active Learning) Pipeline Completed Successfully!"
echo "========================================="
