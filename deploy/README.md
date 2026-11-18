# ROCm deployment: AMD MI300X + vLLM

Everything here runs **on the rented GPU box**. The laptop only runs the app code and points `.env` at the box.

## 1. Rent a GPU

Any provider offering AMD Instinct **MI300X** with ROCm drivers installed, e.g. AMD Developer Cloud, TensorWave,
Vultr, Hot Aisle, or Crusoe. One MI300X (192 GB HBM3) fits both services (7–8B LLM + Whisper) with room for
FP16 and FP8 KV cache. Ask for an Ubuntu image with ROCm and Docker preinstalled.

Open ports 8000 (LLM) and 8001 (ASR) to your IP only, or tunnel them:

```bash
ssh -N -L 8000:localhost:8000 -L 8001:localhost:8001 user@gpu-host
```

## 2. Bring up

```bash
git clone <this repo> && cd concall-analyst
amd-smi list                                   # or: rocm-smi   -> must show the MI300X
cat /opt/rocm/.info/version                    # record ROCm version
cp deploy/rocm.env.example deploy/.env         # set HF_TOKEN for Llama; adjust models
docker compose -f deploy/docker-compose.rocm.yml --env-file deploy/.env up -d
docker compose -f deploy/docker-compose.rocm.yml logs -f    # first start downloads weights into MODEL_CACHE
```

Pinned images (record them in results):

| Service | Image |
|---|---|
| `llm`, `asr` | `rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0` |
| `asr-fallback` | `rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.14.0` |

Container flags required by ROCm are set in the compose file: `/dev/kfd`, `/dev/dri`, `group_add: video`,
`ipc: host`, `seccomp=unconfined`.

## 3. Smoke checks

On the laptop (or the box), with `.env` switched to `ENV=rocm` and the two `*_BASE_URL`s pointing at the box:

```bash
python deploy/smoke.py --audio data/audio/<call>.mp3 --gpu-ssh user@gpu-host
```

It checks: `amd-smi`/`rocm-smi` lists the GPU, `/v1/models` responds on both services, one transcription and
one extraction succeed.

### Whisper on vLLM: verify, then fall back if needed

If the `asr` container fails to start or `/v1/audio/transcriptions` returns 404/500:

1. Try `ASR_EXTRA_ARGS=--task transcription` (older vLLM versions need it).
2. Otherwise use the transformers fallback (`asr_server.py`, same API):
   ```bash
   docker compose -f deploy/docker-compose.rocm.yml --env-file deploy/.env stop asr
   docker compose -f deploy/docker-compose.rocm.yml --env-file deploy/.env --profile asr-fallback up -d asr-fallback
   ```
   and set `ASR_BACKEND=transformers` when you record results.

## 4. Record the environment

`validate.py` and `bench.py` copy these variables into every result row. Export them in the laptop's shell or
add them to `.env`:

```bash
GPU_MODEL="AMD Instinct MI300X"
ROCM_VERSION=$(ssh user@gpu-host cat /opt/rocm/.info/version)
IMAGE_TAG=rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0
VLLM_VERSION=0.23.0        # bench.py also reads it from the server's /version
VLLM_FLAGS="--max-num-seqs 64 --max-num-batched-tokens 8192"
ASR_BACKEND=vllm           # or transformers
LLM_PRECISION=fp16         # fp8 when LLM_EXTRA_ARGS has --quantization fp8
```

## 5. Validation matrix (≥ 3 LLMs × 2 precisions, 2 Whisper variants)

For each configuration: edit `deploy/.env`, `up -d --force-recreate <service>`, wait for health, set the
matching `*_MODEL` / `*_PRECISION` in the laptop `.env`, run validate with **the same `--run-id`** so rows accumulate
and the FP16 gate is computed:

| Step | Server change | Command |
|---|---|---|
| Whisper large-v3 | `ASR_MODEL_ID=openai/whisper-large-v3` | `python eval/validate.py --stage asr --audio … --reference … --run-id val` |
| Whisper turbo | `ASR_MODEL_ID=openai/whisper-large-v3-turbo` | same |
| Qwen FP16 | `LLM_MODEL_ID=Qwen/Qwen2.5-7B-Instruct` | `python eval/validate.py --stage llm --text-source reference --reference … --run-id val` |
| Qwen FP8 | `LLM_EXTRA_ARGS=--quantization fp8 --kv-cache-dtype fp8`, `LLM_PRECISION=fp8` | same |
| Mistral FP16 / FP8 | `LLM_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3` | same |
| Llama FP16 / FP8 | `LLM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct` (needs `HF_TOKEN`) | same |
| End to end | default models | `python eval/validate.py --audio … --reference … --run-id val` |

`--text-source reference` isolates LLM accuracy from ASR errors; the end-to-end row measures both together.

## 6. Benchmarks and tuning

```bash
# on the GPU box (so amd-smi sampling is local) with ENV=rocm in .env:
python eval/bench.py --audio data/audio/<call>.mp3 --concurrency 1 5 20 --max-seconds 600
deploy/sweep.sh data/audio/<call>.mp3 600     # max-num-seqs, batched tokens, memory, chunked prefill, TunableOp, FP8
```

Run benchmarks with `--no-enable-prefix-caching` (sweep.sh does): every concurrent call replays the same audio, so
prefix caching would make 20 calls look as cheap as one.

## 7. Tear down

`docker compose -f deploy/docker-compose.rocm.yml down`, copy `runs/` back to the laptop, and release the GPU.
