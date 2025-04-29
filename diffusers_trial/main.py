import random
import torch
import warnings

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.callbacks import SDXLCFGCutoffCallback
from PIL import Image
from util.diffuser_utils import (
    setup_device,
    finish_pipeline,
    setup_output_directory,
    empty_cache,
)

# 警告を抑制
warnings.filterwarnings("ignore", category=UserWarning, module="diffusers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")


# -------------------- 出力設定 --------------------

save_raw: bool = False
# save_raw = True  # スケール前の画像を保存する時はアンコメント

seed = random.randint(0, 2**32 - 1)
# seed: int = 293115620  # シード値固定してプロンプトを試験する時はアンコメント
print(f"Generated seed: {seed}")

batch_size: int = 1
iteration: int = 3

base_size = (1024, 1024)
scale_enhance_tuple_list = [
    (1600, 1600, 0.1),
]

steps: int = 40
cutoff_step = 20
guidance: int = 8
clip_skip: int = 2

save_dir: str = "result"

# -------------------- モデル --------------------

# https://civitai.com/models/139562/realvisxl-v50
# https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning
base_model: str = "SG161222/RealVisXL_V5.0_Lightning"  # huggingface上のモデル名
scale_model: str = base_model

# https://civitai.com/models/122359/detail-tweaker-xl
detail_lora: str = "model/sdxl/lora/add-detail-xl.safetensors"  # ローカルパス
detail_lora_weight: float = 1.4

# -------------------- プロンプト --------------------

base_prompts_dict = {
    "prompts": (
        # サンプルプロンプト1
        "photorealistic,sakura cream milk ice shake frappe, p1nk1r1fl0wers, made out of iridescent flower petals, high quality, masterpiece,"
        # サンプルプロンプト2
        # "photorealistic,graceful mermaid sits on a rock,playful dolphins swim nearby, coral reef, colorful fish, sunlight through the water, shimmering patterns on the sandy ocean floor,"
    ),
    "negative_prompts": (
        "(malformed),(extra finger:2),(malformed hands:2),(malformed legs:2),(malformed arms:2),(missing finger:2),(malformed feet:2),(extra_legs),"
        "low resolution,worst quality,low quality,normal quality,"
        "mirror, reflection, extra person, text, watermark, logo, caption,label, letters, writing,monochrome,"
    ),
    "loras": {
        detail_lora: detail_lora_weight,
    },
}
scale_prompts_dict = base_prompts_dict

# -------------------- メイン処理 --------------------

if __name__ == "__main__":

    # 保存先ディレクトリ初期化
    setup_output_directory(save_dir)
    image_count: int = 0

    # パイプライン初期化
    device, dtype = setup_device()
    # huggingfaceからベースモデルをダウンロード > from_pretrained()
    # ローカルファイルをロード > from_sigle_file()
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
    ).to(device)
    scale_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        scale_model,
        torch_dtype=dtype,
    ).to(device)

    for i in range(iteration):

        # -------------------- 生成前セットアップ --------------------

        base_prompts = base_prompts_dict["prompts"]
        base_loras = base_prompts_dict["loras"]
        negative_prompts = base_prompts_dict["negative_prompts"]

        scale_prompts = scale_prompts_dict["prompts"]
        scale_loras = scale_prompts_dict["loras"]
        scale_negative_prompts = scale_prompts_dict["negative_prompts"]

        finish_pipeline(base_pipe, lora_weight_dict=base_loras)
        finish_pipeline(scale_pipe, lora_weight_dict=scale_loras)

        empty_cache()

        BASE_GENERATOR = torch.Generator(device=device).manual_seed(seed + image_count)
        SCALE_GENERATOR = torch.Generator(device=device).manual_seed(
            seed + image_count + 9999
        )

        # -------------------- イメージ生成 - ベース --------------------

        base_results = base_pipe(
            prompt=base_prompts,
            negative_prompt=negative_prompts,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=base_size[0],
            height=base_size[1],
            generator=BASE_GENERATOR,
            num_images_per_prompt=batch_size,
            callback_on_step_end=SDXLCFGCutoffCallback(
                cutoff_step_index=cutoff_step,
                cutoff_step_ratio=None,
            ),
            clip_skip=clip_skip,
        )

        image = base_results.images[0]
        if save_raw:
            IMAGE_PATH = f"{save_dir}/{image_count:03}_raw.png"
            image.save(IMAGE_PATH)

        empty_cache()

        # -------------------- イメージ生成 - スケール --------------------

        for k, scale_enhance_tuple in enumerate(scale_enhance_tuple_list):
            scale_width, scale_height, enhance_strength = scale_enhance_tuple

            image = image.resize(
                (scale_width, scale_height),
                resample=Image.Resampling.LANCZOS,
            )
            image = scale_pipe(
                prompt=scale_prompts,
                negative_prompt=scale_negative_prompts,
                image=image,
                strength=enhance_strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=SCALE_GENERATOR,
                callback_on_step_end=SDXLCFGCutoffCallback(
                    cutoff_step_index=cutoff_step,
                    cutoff_step_ratio=None,
                ),
            ).images[0]

            if save_raw and scale_enhance_tuple != scale_enhance_tuple_list[-1]:
                IMAGE_PATH = f"{save_dir}/{image_count:03}_scale_{k}.png"
                image.save(IMAGE_PATH)

            empty_cache()

        # イメージ保存
        final_image_path = f"{save_dir}/{image_count:03}.png"
        image.save(final_image_path)
        image_count += 1
