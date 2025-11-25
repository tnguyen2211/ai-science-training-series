## Part 1: Cerebras CS-3 Batch Size Comparison

### Experimental Results
I executed the training job for 200 steps using two different batch sizes: **1024** (Baseline) and **512**.

| Experiment | Batch Size | Total Steps | Total Time | Avg Throughput (`GlobalRate`) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 1024 | 200 | 7279.6s | **33.06 samples/sec** |
| **Modified** | 512 | 200 | 4109.4s | **33.28 samples/sec** |

### Output Batch size 1024 (Baseline)

```
2025-11-25 15:07:20,457 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-11-25 15:08:20,467 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-11-25 15:08:40,477 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-11-25 15:08:50,486 INFO:   Poll ingress status: Waiting for all Chief pods to be running, current running: 0/1.
2025-11-25 15:09:00,501 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-11-25 15:09:20,511 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-i8jjmtsgutxp9jak2qqsvd&from=1764082621000&to=now
2025-11-25 15:09:20,521 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-i8jjmtsgutxp9jak2qqsvd&from=1764082621000&to=now
2025-11-25 15:09:20,626 INFO:   Preparing to execute using 1 CSX
2025-11-25 15:10:03,927 INFO:   About to send initial weights
2025-11-25 15:10:23,787 INFO:   Finished sending initial weights
2025-11-25 15:10:23,787 INFO:   Finalizing appliance staging for the run
2025-11-25 15:10:23,800 INFO:   Waiting for device programming to complete
2025-11-25 15:13:51,946 INFO:   Device programming is complete
2025-11-25 15:13:53,504 INFO:   Using network type: ROCE
2025-11-25 15:13:53,505 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2025-11-25 15:13:53,515 INFO:   Input workers have begun streaming input data
2025-11-25 15:13:54,652 INFO:   Appliance staging is complete
2025-11-25 15:13:54,653 INFO:   Beginning appliance run
2025-11-25 15:39:45,243 INFO:   | Train Device=CSX, Step=50, Loss=7.67166, Rate=33.08 samples/sec, GlobalRate=33.02 samples/sec, LoopTimeRemaining=1:18:05, TimeRemaining=1:18:05
2025-11-25 16:05:33,548 INFO:   | Train Device=CSX, Step=100, Loss=6.99835, Rate=33.06 samples/sec, GlobalRate=33.04 samples/sec, LoopTimeRemaining=0:52:16, TimeRemaining=0:52:16
2025-11-25 16:31:21,414 INFO:   | Train Device=CSX, Step=150, Loss=6.54189, Rate=33.08 samples/sec, GlobalRate=33.06 samples/sec, LoopTimeRemaining=0:26:28, TimeRemaining=0:26:28
2025-11-25 16:57:09,317 INFO:   | Train Device=CSX, Step=200, Loss=6.10848, Rate=33.11 samples/sec, GlobalRate=33.06 samples/sec, LoopTimeRemaining=0:00:40, TimeRemaining=0:00:40
2025-11-25 16:57:09,323 INFO:   Saving checkpoint at step 200
2025-11-25 17:06:21,205 INFO:   Saved checkpoint model_dir_llama2_7b/checkpoint_200.mdl
2025-11-25 17:06:37,138 INFO:   Training completed successfully!
2025-11-25 17:06:37,145 INFO:   Processed 204800 training sample(s) in 7279.638811971 seconds.
```
### Output Batch size 512 (Modified)
```
2025-11-25 17:16:23,174 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/20.
2025-11-25 17:16:33,185 INFO:   Poll ingress status: Waiting for job ingress readiness.
2025-11-25 17:16:43,195 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-wf8hwb7ns2nh7lrapdj54z&from=1764090364000&to=now
2025-11-25 17:16:43,206 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-wf8hwb7ns2nh7lrapdj54z&from=1764090364000&to=now
2025-11-25 17:16:43,312 INFO:   Preparing to execute using 1 CSX
2025-11-25 17:17:24,816 INFO:   About to send initial weights
2025-11-25 17:17:45,338 INFO:   Finished sending initial weights
2025-11-25 17:17:45,338 INFO:   Finalizing appliance staging for the run
2025-11-25 17:17:45,351 INFO:   Waiting for device programming to complete
2025-11-25 17:21:38,803 INFO:   Device programming is complete
2025-11-25 17:21:40,388 INFO:   Using network type: ROCE
2025-11-25 17:21:40,389 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2025-11-25 17:21:40,402 INFO:   Input workers have begun streaming input data
2025-11-25 17:21:41,628 INFO:   Appliance staging is complete
2025-11-25 17:21:41,628 INFO:   Beginning appliance run
2025-11-25 17:34:32,885 INFO:   | Train Device=CSX, Step=50, Loss=7.81506, Rate=33.26 samples/sec, GlobalRate=33.19 samples/sec, LoopTimeRemaining=0:38:51, TimeRemaining=0:38:51
2025-11-25 17:47:21,543 INFO:   | Train Device=CSX, Step=100, Loss=6.97130, Rate=33.27 samples/sec, GlobalRate=33.25 samples/sec, LoopTimeRemaining=0:26:01, TimeRemaining=0:26:01
2025-11-25 18:00:10,248 INFO:   | Train Device=CSX, Step=150, Loss=6.65227, Rate=33.29 samples/sec, GlobalRate=33.27 samples/sec, LoopTimeRemaining=0:13:12, TimeRemaining=0:13:12
2025-11-25 18:12:58,547 INFO:   | Train Device=CSX, Step=200, Loss=6.32152, Rate=33.38 samples/sec, GlobalRate=33.28 samples/sec, LoopTimeRemaining=0:00:24, TimeRemaining=0:00:24
2025-11-25 18:12:58,553 INFO:   Saving checkpoint at step 200
2025-11-25 18:22:42,325 INFO:   Saved checkpoint model_dir_llama2_7b_bs_512/checkpoint_200.mdl
2025-11-25 18:22:50,410 INFO:   Training completed successfully!
2025-11-25 18:22:50,417 INFO:   Processed 102400 training sample(s) in 4109.419267056 seconds.
```

### Observations & Reasoning

The throughput remained remarkably stable, changing only negligibly from **33.06** to **33.28 samples/sec** despite halving the batch size. This indicates that the Cerebras CS-3 system maintains consistent processing efficiency per sample across this range.

Because the throughput (samples processed per second) remained constant, the time required to complete a step scaled linearly with the batch size.
* **Batch 1024:** Processing 1024 samples took approx. 31 seconds per step ($1024 / 33$).
* **Batch 512:** Processing 512 samples took approx. 15 seconds per step ($512 / 33$).

**Conclusion:**
For the Llama-7B model in this configuration, the Cerebras CS-3 demonstrates **perfect linear scaling** between batch sizes of 512 and 1024. Unlike traditional GPUs where larger batch sizes often significantly improve saturation and throughput (samples/sec), the WSE-3 architecture appears efficiently utilized at both sizes, delivering a steady stream of processed samples without overhead penalties for the smaller batch size.

## Part 2: SambaNova (Metis) vs. NVIDIA (Sophia) Benchmark

I ran a serial benchmark processing 5 samples from the IMDB dataset using the ALCF Inference Endpoints.

| Backend | Hardware | Total Duration | Avg Throughput |
| :--- | :--- | :--- | :--- |
| **Metis** | SambaNova SN40L | **5.95s** | **34.60 tokens/sec** |
| **Sophia** | NVIDIA A100 (vLLM) | 9.99s | 15.42 tokens/sec |

### Output
```
Starting Benchmark...

--- Testing Metis (SambaNova) (Model: gpt-oss-120b-131072) ---
  [Req 1] Success (50 tokens)
  [Req 2] Success (30 tokens)
  [Req 3] Success (27 tokens)
  [Req 4] Success (55 tokens)
  [Req 5] Success (44 tokens)
  > Duration: 5.95s
  > Throughput: 34.60 tokens/sec

--- Testing Sophia (NVIDIA) (Model: openai/gpt-oss-120b) ---
  [Req 1] Success (31 tokens)
  [Req 2] Success (24 tokens)
  [Req 3] Success (33 tokens)
  [Req 4] Success (32 tokens)
  [Req 5] Success (34 tokens)
  > Duration: 9.99s
  > Throughput: 15.42 tokens/sec

```

### Reasoning & Observations

**1. Metis (SambaNova) Superiority for Low Latency:**
In this single-stream benchmark, **Metis** outperformed Sophia by a factor of **~2.2x**. This result highlights the strength of SambaNova's **Dataflow Architecture** (Reconfigurable Dataflow Units). By physically mapping the neural network graph onto the chip's compute units, the SN40L minimizes the "memory wall" bottleneck (moving data between HBM and cores), allowing for extremely low-latency token generation for individual requests.

**2. Sophia (NVIDIA/vLLM) Characteristics:**
Sophia achieved **15.42 tokens/sec**. While slower in this specific test, it is important to note that Sophia runs on **vLLM**, an engine optimized for **aggregate throughput** (serving hundreds of concurrent users) rather than raw single-user latency. The overhead of the vLLM scheduler (PagedAttention) and the HTTP gateway likely contributed to the higher latency observed in this sequential test. In a high-load production scenario with batched requests, Sophia's performance gap would likely narrow or reverse.

### Conclusion
For applications requiring real-time responsiveness for individual queries (like this benchmark), the **SambaNova (Metis)** backend demonstrated superior performance.
