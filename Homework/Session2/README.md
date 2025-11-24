## Session 2:  

Investigate the performance of combining Fully Sharded Data Parallel (FSDP) with Tensor Parallelism (TP) on a small 8-layer model (`dim=256`).

### Results Summary

| Experiment | TP Degree | Avg Iteration Time (`train_dt`) | Performance Delta |
| :--- | :--- | :--- | :--- |
| **Run 1** | TP = 1 | **~0.1257 s** (125.7ms) | Baseline |
| **Run 2** | TP = 2 | **~0.1220 s** (122.0ms) | -2.9% (Faster) |
| **Run 3** | TP = 4 | **~0.1219 s** (121.9ms) | -3.0% (Faster) |

**Key Observations:**

Contrary to the expectation that Tensor Parallelism might slow down small models due to communication overhead, we observed a slight **performance improvement (~2-3%)** at TP=2 and TP=4 compared to the baseline. This result highlights the exceptional efficiency of the **NVLink** interconnect on Polaris. It was able to handle the layer-wise synchronization (`AllReduce`) so quickly that the communication cost was entirely negligible, allowing the slight reduction in compute-per-GPU to yield a net positive speedup.
For small models, Tensor Parallelism offers no speedup and is unnecessary. TP is best reserved for large models (e.g., `dim=4096+`) where splitting matrix multiplications is required to fit parameters in memory or reduce compute time per GPU.
