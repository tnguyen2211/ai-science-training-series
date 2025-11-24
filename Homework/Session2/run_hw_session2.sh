#! /bin/bash -x
#
#
# Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function for logging
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

# --- Environment Setup ---
module use /soft/modulefiles
module load conda/2025-09-25
conda activate base

export PATH="/opt/pbs/bin:${PATH}"  # workaround
export HF_HOME=./.cache 
export WANDB_DISABLED="true"        # Disable generic logging
export OMP_NUM_THREADS=1            # Good practice to prevent CPU thread contention

# --- Configuration ---
LAYERS=8
DATASET="random"
# We want to test TP=1, TP=2, and TP=4
TP_LIST=(1 2 4)

# --- Execution Loop ---
for TP_DEGREE in "${TP_LIST[@]}"; do
    echo "========================================================"
    echo "Running Experiment: Layers=${LAYERS}, TP=${TP_DEGREE}"
    echo "Start Time: $(timestamp)"
    echo "========================================================"

    # Run the ezpz FSDP+TP example
    # Note: ezpz-launch handles the MPI rank distribution automatically
    ezpz-launch python3 -m ezpz.examples.fsdp_tp \
        --dataset ${DATASET} \
        --n-layers ${LAYERS} \
        --tp ${TP_DEGREE} \
    
    echo "Finished TP=${TP_DEGREE} at $(timestamp)"
    echo "--------------------------------------------------------"
    
    # Optional: sleep briefly to let processes clean up
    sleep 5
done