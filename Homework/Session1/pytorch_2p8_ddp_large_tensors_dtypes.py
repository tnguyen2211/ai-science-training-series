from mpi4py import MPI
import os, socket, time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--src-seq-len", type=int, default=1)
    parser.add_argument("--tgt-seq-len", type=int, default=20)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()

args = parse_args()

# --- Setup ---
dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
selected_dtype = dtype_map[args.dtype]

SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
LOCAL_RANK = int(os.environ.get('PALS_LOCAL_RANKID', 0))

# Standard DDP Init
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.polaris.alcf.anl.gov"
os.environ['MASTER_PORT'] = "2345"

torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=RANK, world_size=SIZE)
torch.cuda.set_device(LOCAL_RANK)
device = torch.device(f'cuda:{LOCAL_RANK}')

# --- Data & Model ---
if RANK == 0:
    print(f"\n{'='*40}")
    print(f" CONFIGURATION: Rank Size {SIZE}")
    print(f" Dims: [Samples={args.num_samples}, Embed={args.embed_dim}]")
    print(f" Dtype: {args.dtype} | Batch: {args.batch_size}")
    print(f"{'='*40}\n", flush=True)

# Generate Data
src = torch.rand((args.num_samples, args.src_seq_len, args.embed_dim), dtype=selected_dtype)
tgt = torch.rand((args.num_samples, args.tgt_seq_len, args.embed_dim), dtype=selected_dtype)

dataset = torch.utils.data.TensorDataset(src, tgt)
sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, num_replicas=SIZE, rank=RANK)
loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

model = torch.nn.Transformer(d_model=args.embed_dim, batch_first=True)
if args.dtype == "fp16": model = model.half()
elif args.dtype == "bf16": model = model.bfloat16()

model = model.to(device)
model = DDP(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss().to(device)

# --- Training Loop with Metrics ---
torch.cuda.reset_peak_memory_stats()
total_start = time.time()

for epoch in range(5): # 5 epochs is enough for profiling
    sampler.set_epoch(epoch)
    
    # 1. Start Timer
    t0 = time.time()
    
    # Training
    for source, targets in loader:
        source, targets = source.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(source, targets)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    
    # 2. Stop Compute Timer
    compute_time = time.time() - t0
    
    # 3. Measure Sync/Wait Time
    # (Barrier ensures we wait for the slowest rank)
    torch.distributed.barrier(device_ids=[LOCAL_RANK])
    total_epoch_time = time.time() - t0
    wait_time = total_epoch_time - compute_time
    
    # 4. Calculate Metrics
    # Global throughput = (Samples per Rank * World Size) / Time
    samples_per_rank = len(dataset) // SIZE
    throughput = (samples_per_rank * SIZE) / total_epoch_time
    mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
    
    if RANK == 0:
        print(f"Epoch {epoch+1}: "
              f"Time={total_epoch_time:.2f}s | "
              f"Compute={compute_time:.2f}s | "
              f"Wait={wait_time:.2f}s | "
              f"Rate={throughput:.0f} samp/s | "
              f"Mem={mem_gb:.2f} GB")

# Cleanup
if RANK == 0:
    print(f"\nTotal Run Time: {time.time() - total_start:.2f}s")
torch.distributed.destroy_process_group()