## Task 2: Fix Rank Counting by using Pure mpi4py

**Method:**
`comm.Split_type(MPI.COMM_TYPE_SHARED)` was used to create a communicator containing only processes on the same physical node. The rank within this communicator served as the `LOCAL_RANK`.

** Key Observations:**
* **Rank Initialization:** The logs confirm correct mapping of Global Ranks (0-3) to Local Ranks (0-3) without using PALS environment variables.
* **Processor Affinity:** The `cpubind:list` output confirms that the MPI launcher correctly pinned processes to specific CPU cores, preventing resource contention.
    * Rank 0: `mask 0x3` (Cores 0-1)
    * Rank 1: `mask 0x300` (Cores 8-9)
    * Rank 2: `mask 0x30000` (Cores 16-17)
    * Rank 3: `mask 0x3000000` (Cores 24-25)
* **Performance Note:** The training loop completed in **4.61s**.

## Task 3 & 6: Large Tensors and Different Dtypes
### Comparison Table

| Metric | **Task 3 (Baseline)** | **Task 6 (Scale-Up)** |
| :--- | :--- | :--- |
| **Embedding Dim** | 1024 | 4096 (**4x larger**) |
| **Data Type** | `fp32` (Float32) | `bf16` (BFloat16) |
| **Dataset Size** | 2048 samples | 10,000 samples (**~5x larger**) |
| **Peak Memory** | **4.05 GB** | **15.78 GB** |
| **Throughput** | ~200 - 384 samp/s | ~340 - 370 samp/s |

**Key Observations**

   Increasing the embedding dimension from 1 024 → 4 096 (4×) and the number of samples from 2 048 → 10 000 (≈5×) results in roughly **4× higher peak GPU memory** (4 GB → 15.8 GB). This demonstrates the quadratic memory impact of larger feature dimensions in transformer-style models.
   By switching from `fp32` to `bf16`, we halved the memory required per parameter, fitting the large model into **15.78 GB** (just under the 16GB limit), whereas `fp32` would have caused an Out-Of-Memory (OOM) crash

## Task 4 & 5: Profiling Communication Costs

* **Task 5 (Intra-Node):** 1 Node, 2 Ranks (Uses **NVLink**).
* **Task 4 (Inter-Node):** 2 Nodes, 1 Rank per Node (Uses **Slingshot-11 Network**).

### Results & Analysis

| Metric | Task 4 (Inter-Node) | Task 5 (Intra-Node) | Impact |
| :--- | :--- | :--- | :--- |
| **Total Comm. Time** | **677.21 ms** | **6.97 ms** | **~97x Slower** |
| **Comm. Overhead** | **93.60%** (of GPU time) | **14.40%** (of GPU time) | Severe Bottleneck |
| **Avg Latency** | 30.78 ms | 0.44 ms | High Latency |

**Key Observations:**
The hardware hierarchy on Polaris dictates performance for distributed jobs:
1.  **Intra-node (Task 5):** Communication is negligible (~14% overhead). The GPUs communicate via the high-bandwidth **NVLink** switch (~600 GB/s), resulting in tightly packed compute kernels.
2.  **Inter-node (Task 4):** Communication is the primary bottleneck (~93% overhead). Data must traverse the **Slingshot-11** network cables between cabinets (~25 GB/s), introducing significant latency.



   

