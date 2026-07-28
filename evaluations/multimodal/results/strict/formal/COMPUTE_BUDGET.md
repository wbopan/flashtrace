# Formal multimodal compute budget

Frozen estimate date: 2026-07-27. Hardware: one NVIDIA A100 80GB PCIe.

This estimate is scheduling guidance only. It does not change sample selection,
gates, methods, or endpoints.

## Measured inputs

- Formal VizWiz smoke (`vizwiz-lf-005`): 13.41 seconds for the first greedy
  generation at 261 THINKING tokens and 64 OUTPUT tokens. Strict generation
  additionally performs a repeated greedy generation and two teacher-forced
  OUTPUT-only log-probability forwards.
- Wiki-VISA strict n=18, 2MP: mean first-generation time 2.66 seconds.
- VizWiz native n=9, 1MP: mean first-generation time 3.82 seconds.
- Wiki-VISA strict attribution n=18, 2MP: the frozen eight-method panel totals
  115.28 seconds per sample, dominated by Visual IG at 84.68 seconds and
  Visual LOO at 23.64 seconds.
- VizWiz native faithfulness used 36 regions / 5 steps and took about 4.13
  seconds per sample-method. The formal 64-region / 10-step budget is expected
  to be approximately twice that perturbation time before resolution effects.
- The pilot-disjoint formal-pipeline previews provide the directly matched
  estimate used for the live schedule. Wiki n=20 attribution totals 56.23
  seconds per sample across the eight methods (1.87 hours projected to n=120);
  VizWiz n=20 totals 36.28 seconds per sample (1.01 hours projected to n=100).
  The memory-safe AttnLRP implementation and target-row LM-head projection are
  included in these measurements.
- Under the actual 64-region/10-step budget, Wiki n=20 faithfulness totals
  253.71 seconds per sample across all methods (8.46 hours projected to n=120);
  VizWiz n=20 totals 158.50 seconds per sample (4.40 hours projected to n=100).
  These replace the earlier 36-region/5-step extrapolation for scheduling.
- The preregistered low-yield rule was triggered before attribution: a
  deterministic disjoint Wiki batch of 600 (200 per stratum) was frozen in
  addition to the initial 240. During the first 98 candidates, strict
  generation averaged 22.1 wall-clock seconds per candidate including the
  second greedy decode and two log-probability forwards. This revises the
  candidate-generation projection without changing a gate or endpoint.

## Scheduling range

| Work | Estimated A100 hours |
|---|---:|
| E1/E2 generation and deterministic ablations | 7-10 |
| E3 Wiki localization, eight methods | 1.9-2.3 |
| E4 VizWiz attribution and faithfulness | 5.4-6.2 |
| E5 Wiki faithfulness | 8.5-9.5 |
| Retries and smoke checks | 1-3 |
| Total GPU schedule | 23.8-31.0 |

Runs are serialized on one GPU and use resume-safe JSONL outputs. Cheap
Random/Center methods remain in the paired panel but contribute negligible GPU
time. The removed Grad×Attention run is not included and eliminates the pilot's
only observed OOM path. If fixed-seed formal selection overlaps the breadth-first
n=20 preview, response-identical checkpoint reuse can reduce the realized total
by the exact matched rows; the estimate above does not assume that saving.

## Live execution observations

During the seed-31 Wiki generation-ablation audit, the A100 sustained roughly
57--65% SM utilization, 30--35% memory utilization, 32.7 GiB allocated memory,
and 174--190 W board power without thermal throttling. At the 30-candidate
checkpoint, 18 of 60 ablation generations (30%) reached the frozen
1,024-token cap. The observed bottleneck is therefore batch-one autoregressive
decoding of long reasoning traces, not device-memory capacity. A trial with two
concurrent generation processes reached near-full SM utilization but did not
improve aggregate checkpoint throughput, so the formal run remains serialized
to preserve response identity and avoid protocol drift.

During the resumed Wiki attribution run, heavy methods individually sustained
97--100% SM utilization at 28--37 GiB device memory, but a 45-second
end-to-end sample initially averaged only 28.9% GPU utilization. The first 30
new complete samples accumulated 1,683 seconds of method time in 4,357 seconds
of process wall time. The cause was checkpoint amplification: a 449 MiB
canonical JSONL (including roughly 4 MiB of trace metadata per FlashTrace
record) was atomically rewritten after every sample-method pair, followed by a
full collection and CUDA allocator flush.

The runner now writes each pair to an atomic resume journal and compacts the
canonical JSONL once at completion. Full collection is retained after
high-water white-box methods and at sample boundaries, rather than after every
pair. On the same workload, the next 45-second window averaged 93.2% GPU
utilization, with the GPU non-idle for 97.8% of samples; observed throughput
improved from about 145 seconds to 60--65 seconds per new complete sample.
This is an execution-only change: frozen samples, responses, method formulas,
spatial grids, endpoints, and final JSONL schema are unchanged.

The formal Wiki faithfulness run confirms that checkpoint I/O is no longer the
bottleneck. A 20-second live window averaged 95.7% SM utilization, 67.6%
memory-controller utilization, and 286 W board power, with 32.0 GiB of the
80-GiB framebuffer allocated. GPU temperature remained 78--80 C and only the
software power-cap throttle reason was active; no thermal throttle was
observed. Methods whose signed and positive-only region orders coincide take
about 23 seconds per sample-method. AttnLRP and Visual IG take about 46 seconds,
and Visual LOO about 44 seconds, because their distinct positive-only rankings
require a second perturbation curve. The run remains serialized: a second
worker cannot add useful throughput while the device is already compute- and
power-limited, and changing the perturbation batch shape mid-run would weaken
the numerical consistency of the frozen formal matrix.
