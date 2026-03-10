#!/bin/bash
set -euo pipefail

# ==========================================
# Usage:
#   bash SDDAL.sh <beamshape> <lr> <initial_size> <init_only?> <start_round> <end_round> <gpu> <scanner_batch_size> <retrain_frequency> <scan_only?>
#
# IMPORTANT:
#   - start_round is automatically read out from progress.state. Any manual entry will be ignored.
#   - Number of samples already generated except initial samples equals to round（num batch），so start_round is round index
# ==========================================

beamshape=${1:-chair}
lr=${2:-0.0002}
init_size=${3:-100}
init_only=${4:-false}

# start_round
start_round_arg=${5:-1}

end_round=${6:-580}
gpu=${7:-0}
scanner_batch_size=${8:-5}
retrain_freq=${9:-1}
scan_only=${10:-false}

trainer_batch_size=2  # fixed for Trainer.py

# ------------------------------
# Progress state file (ALWAYS auto-resume)
# start_round := round index (batches generated excluding initial)
# ------------------------------
PROGRESS_FILE="Design_${beamshape}/progress.state"
mkdir -p "Design_${beamshape}"

# Default: start from round 1
start_round=1

if [ -f "${PROGRESS_FILE}" ]; then
    # shellcheck disable=SC1090
    source "${PROGRESS_FILE}" || true
    if [ -n "${next_start_round:-}" ]; then
        start_round="${next_start_round}"
    fi
fi

# Sanity clamp
if [ "${start_round}" -lt 1 ]; then
    start_round=1
fi

# If already finished
if [ "${start_round}" -gt "${end_round}" ]; then
    echo "========================================="
    echo " SDDAL: Nothing to run."
    echo " beamshape     : ${beamshape}"
    echo " progress file : ${PROGRESS_FILE}"
    echo " start_round   : ${start_round}"
    echo " end_round     : ${end_round}"
    echo "========================================="
    exit 0
fi

echo "========================================="
echo " SDDAL Pipeline (AUTO-RESUME ENABLED)"
echo " Beamshape              : ${beamshape}"
echo " Learning rate          : ${lr}"
echo " GPU                    : ${gpu}"
echo " Trainer batch size     : ${trainer_batch_size}"
echo " Scanner batch size     : ${scanner_batch_size}"
echo " Initial set size       : ${init_size}"
echo " Retrain frequency      : ${retrain_freq}"
echo " Scan only?             : ${scan_only}"
echo " Init only?             : ${init_only}"
echo "-----------------------------------------"
echo " start_round (auto)     : ${start_round}"
echo " end_round              : ${end_round}"
echo " (start_round arg given : ${start_round_arg}  -> IGNORED)"
echo " progress file          : ${PROGRESS_FILE}"
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

    # init_only
    cat > "${PROGRESS_FILE}" << EOF
beamshape=${beamshape}
init_size=${init_size}
scanner_batch_size=${scanner_batch_size}
last_finished_round=0
next_start_round=1
total_batches_generated=0
timestamp=$(date +"%F %T")
EOF

    echo "------------------------------"
    echo "  init_only pipeline finished."
    echo "  progress state written: ${PROGRESS_FILE}"
    echo "------------------------------"
    exit 0
fi

# --- Regular behavior (init_only=false) ---
# Run Initializer.py only when:
#   - scan_only=false
#   - and progress indicates we haven't initialized (optional)
#
# Only when start_round==1 run initializer.py
# Now start_round is automatically determined： However, when first time running entire SDDAL pipeline with no sample existing yet, then start_round is automatically 1.
# No initializer will be run if start_round is not 1
if [ "${scan_only}" = true ]; then
    echo "------------------------------"
    echo "  scan_only=true → Skipping Initializer.py"
    echo "------------------------------"
elif [ "${start_round}" -eq 1 ]; then
    echo "------------------------------"
    echo "  Running Initializer.py (first run)"
    echo "------------------------------"

    python3 Initializer.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --init_size ${init_size} \
        --vis_path Design_${beamshape}
else
    echo "------------------------------"
    echo "  Skipping Initializer.py (auto-resume from round ${start_round})"
    echo "------------------------------"
fi

# Loop over rounds
for ((round_sampling=${start_round}; round_sampling<=${end_round}; round_sampling++))
do
    echo "------------------------------"
    echo "  Starting Round ${round_sampling}"
    echo "  (This round index == total batches generated excluding initial)"
    echo "------------------------------"

    # Re-train only when round index matches frequency
    # If scan_only=true → skip training completely
    if [ "${scan_only}" = false ]; then
        if (( (round_sampling-1) % retrain_freq == 0 )); then
            echo "------------------------------"
            echo "  Re-training model at round ${round_sampling}"
            echo "  (Training happens every ${retrain_freq} scans)"
            echo "------------------------------"

            python3 Trainer.py \
                --train_data Design_${beamshape} \
                --epochs 15 \
                --batch_size ${trainer_batch_size} \
                --gpu ${gpu} \
                --lr ${lr} \
                --step_size 2 \
                --seed 123 \
                --pth_name Design_${beamshape}/models/QuantUNetT_${beamshape}
        else
            echo "  Skipping training at this round (waiting for next frequency point)"
        fi
    else
        echo "  scan_only=true → training skipped."
    fi

    # Run Scanner every round (adds new samples to the dataset)
    python3 Scanner.py \
        --beamshape ${beamshape} \
        --gpu ${gpu} \
        --batch_size ${scanner_batch_size} \
        --pth_name QuantUNetT_${beamshape} \
        --round_sampling ${round_sampling} \
        --vis_path Design_${beamshape}

    # ------------------------------
    # Update progress AFTER a successful scan
    # Here "generated batches excluding initial" == round_sampling
    # So next_start_round is round_sampling + 1
    # ------------------------------
    next_round=$((round_sampling + 1))

    cat > "${PROGRESS_FILE}" << EOF
beamshape=${beamshape}
init_size=${init_size}
scanner_batch_size=${scanner_batch_size}
last_finished_round=${round_sampling}
next_start_round=${next_round}
total_batches_generated=${round_sampling}
timestamp=$(date +"%F %T")
EOF

    echo "  Progress updated: last_finished_round=${round_sampling}, next_start_round=${next_round}"
done

# Zernike coefficients statistics
echo "========================================="
echo "   Zernike coefficient statistics in progress…"
echo "========================================="
python3 zernike_statistics.py --beamshape ${beamshape} --init_size ${init_size}

echo "========================================="
echo "   SDDAL Pipeline Completed Successfully!"
echo "   Progress state: ${PROGRESS_FILE}"
echo "========================================="