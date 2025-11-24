from mpi4py import MPI
import os, socket, time
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Task 5: Profiling with Dynamic Args")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--src-seq-len", type=int, default=1)
    parser.add_argument("--tgt-seq-len", type=int, default=20)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="./traces/task_4_5")
    return parser.parse_args()

args = parse_args()

#activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]

# DDP: Set environmental variables used by PyTorch
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID')
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.polaris.alcf.anl.gov"
os.environ['MASTER_PORT'] = str(2345)
print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}.{MASTER_ADDR}")

# DDP: initialize distributed communication with nccl backend
torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=int(RANK), world_size=int(SIZE))

# DDP: pin GPU to local rank.
torch.cuda.set_device(int(LOCAL_RANK))
device = torch.device('cuda')
torch.manual_seed(0)

# Setup Data and Model
dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
selected_dtype = dtype_map[args.dtype]

src = torch.rand((args.num_samples, args.src_seq_len, args.embed_dim), dtype=selected_dtype)
tgt = torch.rand((args.num_samples, args.tgt_seq_len, args.embed_dim), dtype=selected_dtype)
dataset = torch.utils.data.TensorDataset(src, tgt)
sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, num_replicas=SIZE, rank=RANK, seed=0)
loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

model = torch.nn.Transformer(d_model=args.embed_dim, batch_first=True)
if args.dtype == "fp16": model = model.half()
elif args.dtype == "bf16": model = model.bfloat16()
    
# DDP: scale learning rate by the number of GPUs.
optimizer = torch.optim.Adam(model.parameters(), lr=(0.001*SIZE))
criterion = torch.nn.CrossEntropyLoss()
model.train()
model = model.to(device)
criterion = criterion.to(device)
# DDP: wrap the model in DDP
model = DDP(model)

## No compiling
#model = torch.compile(model)

activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)
prof = profile(activities=activities, record_shapes=True, schedule=schedule, profile_memory=True)
if RANK == 0:
    print(f"Starting Profiling Run: 2 Nodes, 1 Rank/Node (Total {SIZE} Ranks)")

prof.start()

start_t = time.time()
for epoch in range(10):
    if RANK == 0:
        print(epoch)
    # DDP: set epoch to sampler for shuffling
    sampler.set_epoch(epoch)

    for source, targets in loader:
        source = source.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        output = model(source, targets)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()
        prof.step()

if RANK == 0:
    print(f'total train time: {time.time() - start_t:.2f}s', flush=True)

prof_timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
print(prof_timestamp)
os.makedirs(args.trace_dir, exist_ok=True)
prof.export_chrome_trace(f"{args.trace_dir}/cuda_pt_2p8-{RANK}-of-{SIZE}.json")
output_path = f"{args.trace_dir}/cuda_pt_2p8_self_cuda_time_total-{RANK}-of-{SIZE}.txt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    f.write(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=-1))

#prof.export_chrome_trace(f"/lus/flare/projects/datasets/softwares/training/atpesc_2025_aiml_profiling/for_atpesc/traces/pytorch_2p8/xpu_compile_train_pt_2p8-{RANK}-of-{SIZE}_"+prof_timestamp+".json") 

# DDP: cleanup
torch.distributed.destroy_process_group()
