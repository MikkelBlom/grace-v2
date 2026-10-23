# Prefill and the llama.cpp prompt cache (fixed prefix + router on a second slot)

Desk research, 2026-10-23. Nothing was run. Grace's flags (scripts/grace.ps1):
`-ngl 99 -c 32768 -np 2 -fa on -ctk q8_0 -ctv q8_0 --reasoning off --swa-full`, plus
`--mmproj ... -ub 1120` when the projector exists. Build b11539.

## 1. What it is
llama-server keeps the KV cache of each slot's last prompt. A new request is sent to a slot; the
longest common prefix (LCP) with that slot's cached tokens is reused and only the rest is
prefilled. Cache misses (changed prefix, slot stolen by another request, SWA checkpoint missing)
mean re-prefilling the whole ~system+tools prompt: this is the up-to-1.5 s miss seen in live run 8.

## 2. Evidence

### Flags (from the server README)
([server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md))
- `--cache-prompt` default on; `cache_prompt` per request.
- `-sps/--slot-prompt-similarity` default 0.10 (0 = off): a request is routed to a slot whose
  cached prompt matches at least that fraction; otherwise the least-recently-used idle slot.
  Older docs/tutorials say 0.5, so check `--help` of b11539.
- `id_slot` in the `/completion` body (OpenAI endpoint passes the same field in recent builds;
  verify) pins a request to a slot; -1 lets the server choose. Later builds still run the LCP
  check when a slot is pinned ([slot pinning note](https://cdn.jsdelivr.net/npm/pi-reasoning-zip@0.7.1/docs/llama-cpp-slot-pinning.md)).
- `--cache-reuse N`: minimum chunk size to reuse via KV *shifting* after an edit earlier in the
  prompt (default 0 = off). Useful only when text in the middle changes but the tail stays.
- `--slot-save-path` + `POST /slots/{id}?action=save|restore|erase`: persist a slot's KV to disk
  and reload it (works as "warm start after restart"); slot state is exact bytes of the model/
  quant/ctx it was made with ([example user](https://pypi.org/project/hearth-llm/)).
- `-cram/--cache-ram` default 8192 MiB: host-RAM prompt cache that keeps evicted slot prompts so
  a returning prompt can be restored instead of recomputed.
- `--context-shift` default disabled (leave it). `-ctxcp/--ctx-checkpoints` default 32,
  `-cms` default 8192 tokens between checkpoints.
- `-kvu/--kv-unified` default "enabled if number of slots is auto"; with explicit `-np 2` each
  slot gets its own 16k as Grace describes.

### Gemma and SWA
Gemma 4 mixes sliding-window and global layers (Gemma 4 31B: 50 SWA layers, 1024-token window,
per an [ik_llama.cpp issue](https://github.com/ikawrakow/ik_llama.cpp/issues/1607); I found no
12B-specific figures). A sliding-window cache cannot roll back to an arbitrary earlier position,
so llama.cpp either keeps context *checkpoints* or, with `--swa-full`, keeps the full-size cache
so prefix reuse works like a normal model, at more KV memory
([explainer](https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache)).
Grace already uses `--swa-full`, which is the right choice for a fixed prefix. Full
re-processing happens ("forcing full prompt re-processing due to lack of cache data", logged
only at `-lv 4`) when no checkpoint/prefix covers the resume position; triggers: edits to earlier
turns, reordered tools, a different slot. That explainer is Qwen-centred and the issues cited
(#22746, #21831) are open "bug-unconfirmed"; Gemma behaviour is inferred, not shown.
Another known weakness: slot choice by LCP-similarity threshold, not best match
([discussion #13606](https://github.com/ggml-org/llama.cpp/discussions/13606)), so a second
request with a different prompt can evict the slot that holds the voice prefix.

### Prefill speed on Blackwell
I found no trustworthy Gemma 4 12B prompt-processing number for an RTX 5090 (laptop or desktop).
Public Gemma 4 5090 pages give only decode (31B ~61 tok/s at 4K; 26B-A4B ~180) and some are
visibly unreliable ([page](https://markaicode.com/benchmarks/gemma-4-llamacpp-benchmark/)).
Phoronix shows the 5090 ~17% over a 4090 for Llama/Mistral 7B prefill
([review](https://www.phoronix.com/review/nvidia-rtx5090-llama-cpp)). Measure it locally with
`llama-bench -p 512,2048 -n 0`. Grace's own numbers (TTFT p50 341 ms with prompt cache) are better
evidence than anything online.

### Flash attention, CUDA graphs, batch size
`-fa on` is already set and is what makes q8_0 V-cache legal. CUDA graphs are on by default in
the CUDA backend (decode); `GGML_CUDA_DISABLE_GRAPHS=1` turns them off and has been a workaround
for MTP-related crashes only. `-ub 1120` (chosen for image tokens) raises the physical batch above
the 512 default, which generally helps long-prefill throughput and costs some compute-buffer VRAM;
text TTFT was measured unchanged (341 vs 343 ms).

### Silent full reprocessing on SWA models: the likely 1.5 s miss
This is the coordinator's hypothesis and it is plausible, not proven for Gemma 4 12B. Mechanism
([explainer](https://particula.tech/blog/prompt-reprocessing-swa-hybrid-models-kv-cache),
Qwen examples; issues [#22746](https://github.com/ggml-org/llama.cpp/issues/22746),
[#21831](https://github.com/ggml-org/llama.cpp/issues/21831),
[#24587](https://github.com/ggml-org/llama.cpp/issues/24587)): on SWA models the server resumes
from a stored context checkpoint; if none sits at or before the common-prefix position it sets
`n_past = 0`, erases every checkpoint and prefills the whole prompt. `--swa-full` should make
this unnecessary because the full-size cache can roll back, but the same explainer says the flag
had no effect when a build logged `n_swa = 0`, and for Gemma 4 nobody has published a check.
Knobs: `--swa-full` (on), `-ctxcp/--ctx-checkpoints` (default 32, ~60-215 MiB each), `-cms`
(default 8192 tokens between checkpoints; lower it toward typical turn length), `--cache-ram`
(host copy of evicted prompts).

**How to confirm in the logs** (llama-server writes to `logs\llama-server.log` via Grace's
script; run once with `-lv 4`, or add `--verbose` temporarily, because the key line is
trace-level):
1. Grep `forcing full prompt re-processing due to lack of cache data`. Any hit during a Grace
   session = a silent full miss; note its timestamp against the trace's slow TTFT.
2. Grep `context checkpoint` (created/restored/erased lines) and `n_swa`/`swa_full` in the model
   load banner: confirm `swa_full = true` took effect for Gemma 4 12B.
3. Per request, compare `slot launch_slot_`/`new prompt, n_ctx_slot, ... task.n_tokens` with
   `n_past` or `cache_n`; the per-request `prompt eval time = ... / N tokens` line shows how many
   tokens were actually prefilled. A voice turn that should add ~50 tokens but prefills thousands
   is the miss. Also record `timings.cache_n`/`prompt_n` from the JSON response in Grace's trace so
   this is visible without log grepping.
4. `--log-prompts-dir` (if present in b11539; check `--help`): diff consecutive prompts for a
   changed earlier token, to separate "prefix mutated" (our prompt) from "slot/checkpoint lost"
   (server).
Also distinguish the other likely cause: the router request landing on the voice slot (fix:
`id_slot` pinning, section 3). A miss that coincides with a router request points to slot
stealing; a miss with no router request in between points to checkpoints or prefix mutation.

## 3. Fit for Grace
Everything above is available on Windows/CUDA in b11539. The design question is slot layout:

1. **Pin slots.** Voice reply requests always `id_slot: 0`; the catalog/router request always
   `id_slot: 1`. Each slot then permanently holds exactly one fixed prefix (system+tools vs.
   catalog), so neither evicts the other. Without pinning, the LCP/LRU choice is the likeliest
   cause of the observed misses (hypothesis; check by logging `cache_n`/`prompt_n` in the
   response `timings` per request).
2. **Keep the prefix byte-identical.** Anything per-turn (time, memory, selected tool schemas)
   must come after the fixed block. If the tool schemas selected by the catalog pick are injected
   through the OpenAI `tools` field, Gemma's chat template renders them near the start of the
   prompt, so a changing tool set invalidates everything after it. This needs checking in the
   rendered prompt (`--log-prompts-dir`, or `/apply-template`): it is the single most likely
   structural miss, bigger than any flag.
3. **Warm at start-up**: send one dummy request per slot after `/health` so both prefixes are
   resident before the first utterance. Optional: save slots to `--slot-save-path` and restore on
   restart (saves a cold prefill of a 130-tool prompt; restore only valid for the same
   model/ctx/flags).
4. **Image requests** share slot 0 and add image tokens after the prefix; fine, but a vision call
   during a voice turn must not use the router slot.
5. `--cache-reuse` is probably not needed (the prefix is fixed, not edited), and a Qwen MTP report
   says it silently disables MTP; multimodal builds may also disable it. Skip.

## 4. Expected gain and costs
- Slot pinning + prefix hygiene: removes miss spikes of up to 1.5 s; p95 TTFT improves, p50
  unchanged. Zero VRAM cost, small code change. Highest value item in this file.
- Shrinking tools: prefill is roughly linear in tokens; the one-line catalog and a trimmed system
  prompt only matter on a *miss* or on the first turn, once the prefix is cached. Do it only if
  misses remain. Dropping verbose schema descriptions and `description` repetition also helps
  Gemma's accuracy on near-miss tools (speculative; test).
- Flags: no further flag gains expected from `-fa`/CUDA graphs; maybe 0-10% from `-ub` tuning.
- Risks: `--swa-full` doubles SWA KV memory (already paid); `-cram` takes up to 8 GiB of the
  64 GB RAM.

## 5. How we would test
1. Replay a recorded multi-turn session (the 130-tool prompt, voice and router requests
   interleaved) through llama-server; log `timings.cache_n` and `prompt_n` for each request. Count
   misses with the current settings, then with `id_slot` pinning, then with pinning + prefix
   reordering. Metric: share of requests with `prompt_n` > 200 (miss) and TTFT p95.
2. Run server at `-lv 4`, grep "forcing full prompt re-processing" to catch SWA misses.
3. `llama-bench -p 512,2048 -ub 512,1120` for the prefill number.
Adopt when: miss rate falls to about 0 in steady state and TTFT p95 drops by >= 300 ms with
p50 unchanged.

## 6. Verdict
**Adopt-candidate** (slot pinning, byte-stable prefix with tool schemas after the fixed block,
start-up warm request). Slot save/restore: maybe. `--cache-reuse`, context shift, flag tuning:
not for us.
