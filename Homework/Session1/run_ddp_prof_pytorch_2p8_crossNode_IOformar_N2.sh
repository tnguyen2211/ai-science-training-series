#! /bin/bash -x
#
# Combined Runner for Task 4 (Inter-node) and Task 5 (Intra-node Baseline)

# Timezone & Stamp
export TZ='/usr/share/zoneinfo/US/Central'
tstamp() { date +"%Y-%m-%d-%H%M%S"; }

# Common Environment
module use /soft/modulefiles
module load conda/2025-09-25
conda activate
export DISABLE_PYMODULE_LOG=1

# ==========================================
# PART 1: Task 4 (Inter-node Communication)
# Goal: 2 Ranks total, spread across 2 Nodes (1 Rank per Node)
# ==========================================
echo ">>> Starting Task 4: Inter-node (2 Nodes, 1 Rank/Node)"

N=2              # Total Ranks
PPN=1            # 1 Rank per node (Forces network communication)
EPOCHS=3

# Unique trace directory for Task 4
TRACE_DIR_T4=./traces/pytorch_2p8/task4_inter_${tstamp}

# Affinity: Bind the single rank to core 0
export CPU_AFFINITY="verbose,list:0"

mpiexec -n ${N} -ppn ${PPN} -l --line-buffer --cpu-bind ${CPU_AFFINITY} \
    python pytorch_2p8_ddp_prof.py \
    --epochs ${EPOCHS} \
    --trace-dir ${TRACE_DIR_T4}

echo ">>> Task 4 Complete. Traces saved to ${TRACE_DIR_T4}"
echo "-----------------------------------------------------"

# ==========================================
# PART 2: Task 5 Baseline (Intra-node Communication)
# Goal: 2 Ranks total, both on 1 Node
# ==========================================
echo ">>> Starting Task 5: Intra-node Baseline (1 Node, 2 Ranks)"

N=2              # Total Ranks
PPN=2            # 2 Ranks per node (Uses local NVLink)
EPOCHS=3

# Unique trace directory for Task 5
TRACE_DIR_T5=./traces/pytorch_2p8/task5_intra_${tstamp}

# Affinity: Bind the two ranks to cores 0 and 1
export CPU_AFFINITY="verbose,list:0,1"

# Note: Even though we have 2 nodes allocated, requesting -n 2 -ppn 2 
# will put both ranks on the *first* node automatically.
mpiexec -n ${N} -ppn ${PPN} -l --line-buffer --cpu-bind ${CPU_AFFINITY} \
    python pytorch_2p8_ddp_prof.py \
    --epochs ${EPOCHS} \
    --trace-dir ${TRACE_DIR_T5}

echo ">>> Task 5 Complete. Traces saved to ${TRACE_DIR_T5}"
