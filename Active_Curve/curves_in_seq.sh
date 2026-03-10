#!/bin/bash

wait_until_beam_done () {
    local beam="$1"

    while pgrep -f "python3 train_unet.py .*Design_${beam}" > /dev/null; do
        echo "$(date '+%F %T')  Waiting for ${beam} jobs to finish..."
        sleep 20
    done
}

run_one_beam () {
    local beam="$1"
    local lr="$2"

    echo "=================================================="
    echo "$(date '+%F %T')  Launching beam=${beam}, lr=${lr}"
    echo "=================================================="

    bash TrainSet_curve_sddal.sh "$beam" "$lr"

    echo "$(date '+%F %T')  All ${beam} jobs launched. Now waiting..."
    wait_until_beam_done "$beam"

    echo "$(date '+%F %T')  ${beam} completely finished."
    echo
}

run_one_beam rec 0.0002
run_one_beam chair 0.0002
run_one_beam tear 0.0002
run_one_beam ring 0.0002
run_one_beam hat 0.0002
run_one_beam gaussian 0.0001

echo "$(date '+%F %T')  All beamshapes finished."