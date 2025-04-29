import os
import numpy as np
from PIL import Image
import shutil
import torch

from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
)


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_device():
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Using device: {device}")
    return device, dtype


def finish_pipeline(
    pipe,
    lora_weight_dict,
    scheduler=EulerAncestralDiscreteScheduler,
    embeddings=None,
):
    # Embeddingのロードを実装
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.safety_checker = None
    load_loras(pipe, lora_weight_dict)

    if embeddings:
        for embedding_path, token in embeddings:
            try:
                pipe.load_textual_inversion(embedding_path, token=token)
                print(f"Loaded embedding: {token} from {embedding_path}")
            except Exception as e:
                print(f"Error loading embedding {embedding_path}: {e}")
    load_embeddings(pipe, embeddings)

    return pipe


def load_embeddings(pipe, embeddings):
    if embeddings:
        for embedding_path, token in embeddings:
            try:
                pipe.load_textual_inversion(embedding_path, token=token)
                print(f"Loaded embedding: {token} from {embedding_path}")
            except Exception as e:
                print(f"Error loading embedding {embedding_path}: {e}")


def setup_output_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name, exist_ok=True)


def load_loras(pipe, lora_weight_dict):
    adapter_names = []
    adapter_weights = []

    # 入力されたLoRAをパス→名前に変換
    REQUESTED_LORA_PATHS = list(lora_weight_dict.keys())
    REQUESTED_LORA_NAMES = [
        os.path.basename(path).split(".")[0] for path in REQUESTED_LORA_PATHS
    ]

    # すでにロード済みのLoRA名を取得
    CURRENT_ADAPTERS = pipe.get_list_adapters()
    CURRENT_LORA_NAMES = CURRENT_ADAPTERS.get("unet", [])

    # 新しくロードすべきLoRA
    NEW_LORA_NAMES = [
        name
        for i, name in enumerate(REQUESTED_LORA_NAMES)
        if name not in CURRENT_LORA_NAMES
    ]

    # アンロードすべき古いLoRA
    OLD_LORA_NAMES = [
        name for name in CURRENT_LORA_NAMES if name not in REQUESTED_LORA_NAMES
    ]

    # 古いLoRAをアンロード
    for name in OLD_LORA_NAMES:
        pipe.unload_lora_weights(adapter_name=name)

    # 新しいLoRAをロード
    for i, name in enumerate(REQUESTED_LORA_NAMES):
        print(f"Loading LoRA: {name}")
        path = REQUESTED_LORA_PATHS[i]
        weight = lora_weight_dict[path]

        # 新規LoRAの場合はロード
        if name in NEW_LORA_NAMES:
            try:
                pipe.load_lora_weights(path, adapter_name=name, prefix=None)
            except Exception as e:
                print(f"Loading LoRA {path} failed: {e}")
                continue

        # 既存または新規にロードしたLoRAの重みを設定
        adapter_names.append(name)
        adapter_weights.append(weight)

    # アダプター設定を適用
    if adapter_names:
        print(f"Setting adapters: {adapter_names} with weights {adapter_weights}")
        pipe.set_adapters(adapter_names, adapter_weights)

    return pipe
