import subprocess
import modal

app = modal.App("layerstream-inference")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
logs = modal.Volume.from_name("layerstream-logs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        # add your other deps here
    )
    .add_local_dir("layerstream", remote_path="/root/layerstream")
)

@app.function(
    image=image,
    gpu="A100-40GB",   # or "L40S", "A10", "H100", etc.
    timeout=60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/logs": logs,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret")
    ],
)
def run_layerstream(
    model: str = "meta-llama/Llama-2-13b-chat-hf",
    mode: str = "layered",
    prompt: str = "Hello, I am a language model,",
    max_new_tokens: int = 64,
    dtype: str = "float16",
    buffer_depth: int = 2,
    max_gpu_kv_gb: float = 2.0,
    max_seq_len: int = 2048,
):
    cmd = [
        "python",
        "-m",
        "layerstream.main",
        "--model", model,
        "--mode", mode,
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--dtype", dtype,
        "--buffer-depth", str(buffer_depth),
        "--max-gpu-kv-gb", str(max_gpu_kv_gb),
        "--max-seq-len", str(max_seq_len),
        "--log-dir", "/root/logs",
    ]

    result = subprocess.run(
        cmd,
        cwd="/root",
        text=True,
        capture_output=True,
        check=False,
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.local_entrypoint()
def main(
    prompt: str = "Hello, I am a language model,",
    model: str = "meta-llama/Llama-2-13b-chat-hf",
    mode: str = "layered",
    max_new_tokens: int = 64,
):
    result = run_layerstream.remote(
        model=model,
        mode=mode,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    print("STDOUT:")
    print(result["stdout"])

    if result["stderr"]:
        print("STDERR:")
        print(result["stderr"])