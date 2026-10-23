# Laptop hardware: power limits, sustained load, NPU and iGPU offload

Desk research 2026-10-23. Few laptop-specific LLM measurements exist publicly; much below is
general knowledge, labelled as such. Everything here needs a local `nvidia-smi`/`llama-bench` check.

## 1. What it is
The RTX 5090 Laptop (24 GB GDDR7, 896 GB/s per NVIDIA spec; general knowledge) runs inside a shared
power and thermal budget: maximum graphics power varies by chassis, and Dynamic Boost moves watts
between CPU and GPU. Windows power mode and AC/battery change the budget.

## 2. Evidence
- Power limits vary per chassis: StorageReview saw a Razer Blade 16 5090 at 135 W in performance
  mode against a 150 W maximum, while gaming chassis reach 175 W (150 W + 25 W Dynamic Boost),
  as in Notebookcheck's [5090 Laptop analysis](https://www.notebookcheck.net/Nvidia-GeForce-RTX-5090-Laptop-Analysis-Faster-but-there-is-more-to-come.988519.0.html)
  via [memeburn](https://memeburn.com/best-laptop-for-running-ai-locally/) (secondary).
- LLM impact ([computingforgeeks](https://computingforgeeks.com/best-laptop-for-local-llm/),
  secondary, reporting an HP Omen Max 16): decode at 140-160 W is within 1-4 % of 175 W, but about
  -28 % in the 95 W mode; prompt processing (compute-bound) falls with every watt lost. On battery,
  StorageReview measured -40 to -47 % tok/s on the Blade 16.
  Verified at source (I re-read the article): the HP Omen Max 16 figures (140-160 W within 1-4 %
  of 175 W; "95W power mode" drops about 28 %) are reported benchmark results from the llama.cpp
  CUDA scoreboard, not the author's estimates, but the article does **not** name the model or quant
  for them (its own llama-bench table uses Llama 2 7B Q4_0), and I did not open the scoreboard
  itself. So: plausible and consistent with bandwidth-bound decode, single-chassis, unverified on
  Grace's laptop. The battery figure is StorageReview's (Phi 163 to 99, Mistral 126 to 67 tok/s).
  Interpretation: decode is mostly memory-bound so mild power caps cost almost nothing; below about
  100 W the memory clocks/GPU clocks drop and it hurts.
- Dynamic Boost: when llama-server is fully offloaded the CPU is near idle, so the GPU should get
  most of the shared budget regardless of Windows power mode (general knowledge, search found
  nothing that tests this). Power mode matters when CPU work competes (STT on CPU, embeddings,
  browser, YouTube decoding) because the CPU then pulls budget and heat from the same heat pipes.
  Verify with `nvidia-smi --query-gpu=power.draw,power.limit,clocks.sm,clocks.mem,clocks_throttle_reasons.active`.
- Grace-specific hint: decode of 50 tok/s with a 7 GB model is about half of the bandwidth ceiling
  (about 128 tok/s); the gap is not just power. Candidates: `-np 2`, SWA overhead, launch overhead,
  clocks. A `llama-bench` run at `-p 0 -n 128` with no server would separate engine from laptop.
- Undervolting/power capping: general practice (not sourced): capping to about 120-140 W or a
  +/- clock-offset curve in MSI Afterburner/NVIDIA App typically loses a few percent of throughput
  on bandwidth-bound loads and cuts temperature and fan noise noticeably. Laptop BIOS often locks
  voltage; Afterburner curve editing may or may not work on this chassis (unknown).
- Longevity: silicon degrades with heat and voltage; sustained inference at 70-80 C GPU is within
  normal operating range (NVIDIA's throttle point is higher, about 87 C for laptop parts; general
  knowledge). Bigger practical risks: fan wear, battery swelling when kept at 100 % on AC (use the
  vendor's battery-care / 80 % cap), hot VRAM (check memory junction temp, which `nvidia-smi` shows
  as `temperature.memory` on some drivers), and thermal paste pump-out over years. No quantitative
  study found for LLM duty cycles.
- Overnight LoRA training: compute-bound, so unlike inference it scales with power. A cap of
  about 125-140 W typically loses far less than proportional throughput in perf/W terms (general
  knowledge) and keeps temps and noise down; measure it-per-watt with a 5-minute training step test
  at 3 power limits.

## 3. NPU and Arc 140T iGPU: what they can take off the RTX
- OpenVINO 2026.0 broadened NPU model support (Qwen3-Embedding-0.6B, small Qwen2.5 models,
  MiniCPM-o) per [Phoronix](https://www.phoronix.com/news/Intel-OpenVINO-2026.0-Released) and the
  [NPU GenAI docs](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html).
  NPU prefers symmetric INT4 and static shapes; Grace already found the NPU cannot run Qwen3.5's
  hybrid attention.
- Whisper-family on NPU: OpenVINO GenAI's Whisper pipeline runs on NPU but needs models exported
  with `--disable-stateful`; one Lunar Lake report claims about 50x real-time for whisper-base.en
  INT8 (community RFC,
  [elizaOS #7633](https://github.com/elizaOS/eliza/issues/7633), not on Core Ultra 9 / Arrow
  Lake). Relevance to Grace is limited: STT is "edda" (Danish), not a Whisper model, so it would need
  an OpenVINO export and a parity check on Danish WER.
- Serving: [OpenArc](https://github.com/FXeio/OpenArc) (secondary/community) wraps OpenVINO for
  LLM, Whisper and embeddings with OpenAI endpoints on Intel NPU/GPU/CPU.
- Realistic offload candidates, already partly done: the yes/no checker (JevK5 4B INT8 on the iGPU,
  measured 208 ms p50, 3.5 GB), embeddings (harrier-270m on CPU, 51 ms; NPU move saves CPU, not
  latency), VAD (tiny; CPU is fine), maybe a small STT or wake-word. The iGPU shares system RAM
  bandwidth (LPDDR5x, far below 896 GB/s) so it cannot take over a 12B; the NPU is power-efficient,
  not fast. Main benefit: keeps the RTX decode from being slowed (Grace measured Gemma decode
  50 to 33 tok/s while a 4B checker ran on the same GPU), plus frees watts for the dGPU.
- Caveat on the iGPU: on a hybrid laptop the Arc iGPU and CPU also share the package power budget
  with Dynamic Boost, so heavy iGPU work can trim CPU clocks, but not the dGPU's watts directly.
- YouTube/video decode: Chrome/Edge on Windows decodes video with the iGPU (Intel Quick Sync) or
  NVDEC on the dGPU depending on the per-app graphics preference (Settings > System > Display >
  Graphics). If it lands on the RTX it costs a small amount of VRAM and some bandwidth and can
  perturb latency while Grace runs. Fix by pinning the browser to "Power saving" (iGPU). This is a
  configuration change for Mikkel to decide; not done here.

## 4. Expected gain, risks
- Power-capping the RTX to about 140-150 W: -0 to -4 % decode (secondary figures), lower temps,
  lower fan noise (mic and echo friendliness), better longevity. Free to try, reversible.
- Pinning browser video to the iGPU: removes a plausible source of latency jitter in tails
  (unmeasured).
- Moving more side models to NPU: gain mostly CPU headroom; needs per-model export work; low
  priority.
- Risks: NPU driver/OpenVINO version churn; undervolting instability (crash mid-run); Windows power
  plans changing after updates.

## 5. How we would test it
1. Baseline: `llama-bench` (pp512, tg128) on the Gemma 4 12B QAT Q4_0 with no server, AC, Best
   Performance, log `nvidia-smi` power/clocks/throttle reasons once per second.
2. Repeat with GPU power limit 175/150/130/110 W (`nvidia-smi -pl` may be locked on laptops; else
   the vendor control app), and Windows power mode Balanced vs Best Performance, 3 runs each.
3. Repeat step 1 with a 1080p YouTube video playing, once with the browser on the iGPU and once on
   the RTX; then run the 20-request TTFT harness and compare p95.
4. For training: tokens/s per watt at the same limits for a short LoRA step.
Adopt a cap if decode loses < 5 % and TTFT p95 does not move; adopt the browser pinning if p95
improves with video on.

## 6. Verdict
- Power cap / power mode: **maybe** (cheap measurement; likely small effect on inference, useful
  for training and noise).
- Browser video on the iGPU: **adopt-candidate** as a measured check, since it costs nothing.
- NPU/iGPU offload of more models: **not for us** now; the iGPU/NPU already hold the useful small
  jobs and cannot carry the main model.
