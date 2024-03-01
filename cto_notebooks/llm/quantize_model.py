# %%
# Setup paths and quantization mode
import subprocess
from typing import List

from cto_notebooks.utils.config import CONFIG

LLAMA_DIR = CONFIG.cache_dir.joinpath("llama_cpp")
MODEL_PATH = CONFIG.model_dir.joinpath("em_german_leo_mistral_lora_cardiode")
GGUF_PATH = MODEL_PATH.as_posix() + ".gguf"
QUANTIZATION = "Q5_K_M"
QUANTIZED_PATH = GGUF_PATH.replace(".gguf", f"_{QUANTIZATION}.gguf")


def call(cmd: List[str], cwd: str = None) -> None:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd)  # noqa: S603
    for line in p.stdout:
        print(line, end="")


# %%
# Clone llama.cpp into cache
call(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "b2297",
        "https://github.com/ggerganov/llama.cpp.git",
        LLAMA_DIR,
    ]
)


# %%
# Install requirements and build binaries
call(["pip", "install", "-r", "requirements.txt"], LLAMA_DIR)
call(["make"], LLAMA_DIR)

# %%
# Convert model to gguf
call(
    [
        "python",
        "convert.py",
        MODEL_PATH,
        "--outtype",
        "f32",
        "--vocab-type",
        "hfft",
        "--outfile",
        GGUF_PATH,
    ],
    LLAMA_DIR,
)

# %%
# Quantize gguf model
call(["./quantize", GGUF_PATH, QUANTIZED_PATH, QUANTIZATION], LLAMA_DIR)

# %%
