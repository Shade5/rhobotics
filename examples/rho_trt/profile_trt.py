"""Kernel-level profile of one CUDA-graph replay of the sampler, and a per-layer profile of the prefix engine."""

import collections
import pathlib

import common
import tensorrt as trt
import torch
import trt_runtime

engine_dir = common.CACHE_DIR / pathlib.Path(common.CHECKPOINT).name
policy = common.load_policy()
flow = policy.model.flow_model
io = torch.load(engine_dir / "sample_io.pt")
sampler = trt_runtime.Sampler(flow, engine_dir)
images, ids, state, noise = (io[k].cuda() for k in ("images", "ids", "state", "noise"))
sampler.run(images, ids, state, noise)
g = sampler._graphs[(ids.shape[0], False)]  # noqa: SLF001

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(5):
        g.graph.replay()
    torch.cuda.synchronize()
kernels = collections.Counter()
counts = collections.Counter()
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        name = e.name
        kind = "triton (step)" if name.startswith("triton") else ("trt/cublas/other")
        kernels[kind] += e.device_time / 5
        counts[kind] += 1 / 5
for kind, us in kernels.items():
    print(f"{kind:20s} {us / 1e3:7.2f} ms over {counts[kind]:.0f} kernels")
top = collections.Counter()
for e in prof.events():
    if e.device_type == torch.autograd.DeviceType.CUDA:
        top[e.name[:90]] += e.device_time / 5
print("top kernels:")
for name, us in top.most_common(25):
    print(f"  {us / 1e3:6.3f} ms  {name}")


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.times = collections.Counter()

    def report_layer_time(self, name, ms):
        self.times[name] += ms


p = Profiler()
ctx = sampler.prefix.context
ctx.profiler = p
for _ in range(10):
    sampler.prefix.run(torch.cuda.current_stream(), images=g.images, text=g.text, **g.kv)
torch.cuda.synchronize()
total = sum(p.times.values()) / 10
print(f"prefix engine layers: {total:.2f} ms total")
for name, ms in p.times.most_common(30):
    print(f"  {ms / 10:6.3f} ms  {name[:150]}")
