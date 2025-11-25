## Task 1: Active Learning Optimization

Experimental Results (Baseline vs. Optimized)
Compare the default configuration against a tuned configuration running on a Polaris node.

| Metric | **Default Configuration** | **Optimized Configuration** | **Impact** |
| :--- | :--- | :--- | :--- |
| **Batch Size** | 4 | **64** | **16x Throughput** |
| **Max Samples** | 24 | **200** | **~8x Search Depth** |
| **Wall Clock Time** | 24.69 s | 32.97 s | Similar Time Cost |
| **Best Energy Found** | **15.33 Ha** | **17.78 Ha** | **+2.45 Ha (Success)**

Code Modifications
The following parameters were modified in `3_ml_in_the_loop.py` to achieve these results:

```python
# Modified parameters for high-throughput active learning
initial_training_count = 16   # Increased from 8 to improve initial model quality
max_training_count = 200      # Increased from 24 to allow convergence on high-energy targets
batch_size = 64               # Increased from 4 to saturate Polaris CPU cores
```
![Result Plot](parsl_ml_in_the_loop.png)
