#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib"]
# ///
"""Generate GPU scaling chart from test results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent / "charts"
OUT_DIR.mkdir(exist_ok=True)
OBSIDIAN_DIR = Path.home() / "Obsidian" / "Basic Memory" / "Reports" / "cascade" / "attachments"
OBSIDIAN_DIR.mkdir(parents=True, exist_ok=True)

# GPU scaling test data (RTX 3090)
batch_sizes = [1, 4, 16, 45, 64, 128, 256, 512, 1024, 2048]
kernel_times = [9.20, 11.72, 26.65, 31.76, 31.98, 31.99, 32.65, 51.39, 51.98, 52.88]
sims_per_sec = [0.11, 0.34, 0.60, 1.42, 2.00, 4.00, 7.84, 9.96, 19.70, 38.73]
time_per_sim_ms = [9199, 2929, 1666, 706, 500, 250, 128, 100, 51, 26]

# CPU Mojo baseline: 0.45s per sim = 450ms
cpu_per_sim_ms = 450

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Chart 1: Throughput vs Batch Size
ax = axes[0]
ax.plot(batch_sizes, sims_per_sec, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='GPU')
ax.axhline(y=2.24, color='#3498db', linestyle='--', linewidth=1.5, label='CPU Mojo (2.24 sims/s)')
ax.axhline(y=0.59, color='#e74c3c', linestyle='--', linewidth=1.5, label='Python (0.59 sims/s)')
ax.set_xlabel('Batch Size (# simulations)', fontsize=12)
ax.set_ylabel('Throughput (sims/sec)', fontsize=12)
ax.set_title('GPU Throughput vs Batch Size', fontsize=13, fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(batch_sizes)
ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=8, rotation=45)

# Chart 2: Time per Simulation
ax = axes[1]
ax.plot(batch_sizes, time_per_sim_ms, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='GPU')
ax.axhline(y=cpu_per_sim_ms, color='#3498db', linestyle='--', linewidth=1.5, label='CPU Mojo (450ms)')
ax.axhline(y=1690, color='#e74c3c', linestyle='--', linewidth=1.5, label='Python (1690ms)')
ax.set_xlabel('Batch Size (# simulations)', fontsize=12)
ax.set_ylabel('Time per Simulation (ms)', fontsize=12)
ax.set_title('Amortized Time per Sim', fontsize=13, fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(batch_sizes)
ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=8, rotation=45)

# Annotate crossover point
ax.annotate('GPU beats CPU\nat batch >= 64', xy=(64, 500), xytext=(200, 1200),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray', ha='center')

# Chart 3: Wall Clock Time vs Batch Size
ax = axes[2]
ax.plot(batch_sizes, kernel_times, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='GPU kernel time')
cpu_sequential = [b * 0.45 for b in batch_sizes]
ax.plot(batch_sizes, cpu_sequential, 's--', color='#3498db', linewidth=1.5, markersize=6, label='CPU Mojo sequential')
ax.set_xlabel('Batch Size (# simulations)', fontsize=12)
ax.set_ylabel('Wall Clock Time (seconds)', fontsize=12)
ax.set_title('Wall Clock: GPU vs CPU Sequential', fontsize=13, fontweight='bold')
ax.set_xscale('log', base=2)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(batch_sizes)
ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=8, rotation=45)

# Annotate
ax.annotate(f'GPU: 53s\nCPU: 921s\n17.4x faster', xy=(2048, 52.88),
            xytext=(400, 600), arrowprops=dict(arrowstyle='->', color='#2ecc71'),
            fontsize=9, color='#1e8449', fontweight='bold')

plt.tight_layout()
fig.savefig(OUT_DIR / 'gpu_scaling.png', dpi=150, bbox_inches='tight')
fig.savefig(OBSIDIAN_DIR / 'cascade-gpu-scaling.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'gpu_scaling.png'}")
print(f"Saved: {OBSIDIAN_DIR / 'cascade-gpu-scaling.png'}")
