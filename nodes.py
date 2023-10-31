"""
@author: receyuki
@title: SD Prompt Reader
@nickname: SD Prompt Reader
@description: ComfyUI node version of the SD Prompt Reader
"""


import os
from datetime import datetime
from itertools import chain

import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

import hashlib
import piexif
import piexif.helper


from nodes import MAX_RESOLUTION
from comfy.cli_args import args
import comfy.samplers
import folder_paths

from .stable_diffusion_prompt_reader.sd_prompt_reader.constants import (
    SUPPORTED_FORMATS,
    MESSAGE,
)
from .stable_diffusion_prompt_reader.sd_prompt_reader.image_data_reader import (
    ImageDataReader,
)

from .__version__ import VERSION as NODE_VERSION
from .stable_diffusion_prompt_reader.sd_prompt_reader.__version__ import (
    VERSION as CORE_VERSION,
)

BLUE = "\033[1;34m"
CYAN = "\033[36m"
RESET = "\033[0m"


def output_to_terminal(text: str):
    print(f"{RESET+BLUE}" f"[SD Prompt Reader] " f"{CYAN+text+RESET}")


output_to_terminal("Node version: " + NODE_VERSION)
output_to_terminal("Core version: " + CORE_VERSION)


class SDPromptReader:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "parameter_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 255, "step": 1},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "IMAGE",
        "MASK",
        "POSITIVE",
        "NEGATIVE",
        "SEED",
        "STEPS",
        "CFG",
        "WIDTH",
        "HEIGHT",
        "FILE_NAME",
        "SETTINGS",
    )

    FUNCTION = "load_image"
    CATEGORY = "SD Prompt Reader"
    OUTPUT_NODE = True

    def load_image(self, image, parameter_index):
        image_path = folder_paths.get_annotated_filepath(image)
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        file_path = Path(image_path)

        if file_path.suffix not in SUPPORTED_FORMATS:
            output_to_terminal(MESSAGE["suffix_error"][1])
            raise ValueError(MESSAGE["suffix_error"][1])

        with open(file_path, "rb") as f:
            image_data = ImageDataReader(f)
            if not image_data.tool:
                output_to_terminal(MESSAGE["format_error"][1])
                raise ValueError(MESSAGE["format_error"][1])

            seed = int(
                self.param_parser(image_data.parameter.get("seed"), parameter_index)
                or 0
            )
            steps = int(
                self.param_parser(image_data.parameter.get("steps"), parameter_index)
                or 0
            )
            cfg = float(
                self.param_parser(image_data.parameter.get("cfg"), parameter_index) or 0
            )
            width = int(image_data.width or 0)
            height = int(image_data.height or 0)

            output_to_terminal("Positive: \n" + image_data.positive)
            output_to_terminal("Negative: \n" + image_data.negative)
            output_to_terminal("Setting: \n" + image_data.setting)
        return {
            "ui": {
                "text": (image_data.positive, image_data.negative, image_data.setting)
            },
            "result": (
                image,
                mask,
                image_data.positive,
                image_data.negative,
                seed,
                steps,
                cfg,
                width,
                height,
                file_path.stem,
                image_data.setting,
            ),
        }

    @staticmethod
    def param_parser(data: str, index: int):
        data_list = data.strip("()").split(",")
        return data_list[0] if len(data_list) == 1 else data_list[index]

    @classmethod
    def IS_CHANGED(s, image, parameter_index):
        image_path = folder_paths.get_annotated_filepath(image)
        with open(Path(image_path), "rb") as f:
            image_data = ImageDataReader(f)
        return image_data.props

    @classmethod
    def VALIDATE_INPUTS(s, image, parameter_index):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


class SDPromptSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "filename": (
                    "STRING",
                    {"default": "ComfyUI_%time_%seed_%counter", "multiline": False},
                ),
                "path": ("STRING", {"default": "%date/", "multiline": False}),
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                # "model_name_str": ("STRING", {"default": ""}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 10000},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                # "sampler_name_str": ("STRING", {"default": ""}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                # "scheduler_str": ("STRING", {"default": ""}),
                "width": (
                    "INT",
                    {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 1, "min": 1, "max": MAX_RESOLUTION, "step": 8},
                ),
                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),
                "extension": (["png", "jpg", "webp"],),
                "calculate_model_hash": ("BOOLEAN", {"default": False}),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "jpg_webp_quality": ("INT", {"default": 100, "min": 1, "max": 100}),
                "date_format": (
                    "STRING",
                    {"default": "%Y-%m-%d", "multiline": False},
                ),
                "time_format": (
                    "STRING",
                    {"default": "%H%M%S", "multiline": False},
                ),
                "extra_info": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "SD Prompt Reader"

    def save_images(
        self,
        images,
        filename: str = "ComfyUI_%time_%seed_%counter",
        path: str = "%date/",
        model_name: str = "",
        model_name_str: str = "",
        seed: int = 0,
        steps: int = 0,
        cfg: float = 0.0,
        sampler_name: str = "",
        sampler_name_str: str = "",
        scheduler: str = "",
        scheduler_str: str = "",
        width: int = 1,
        height: int = 1,
        positive: str = "",
        negative: str = "",
        extension: str = "png",
        calculate_model_hash: bool = False,
        lossless_webp: bool = True,
        jpg_webp_quality: int = 100,
        date_format: str = "%Y-%m-%d",
        time_format: str = "%H%M%S",
        extra_info: str = "",
        prompt=None,
        extra_pnginfo=None,
    ):
        (
            full_output_folder,
            filename_alt,
            counter_alt,
            subfolder_alt,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            self.prefix_append,
            self.output_dir,
            images[0].shape[1],
            images[0].shape[0],
        )

        results = list()
        for image in images:
            # model_name_str, sampler_name_str, scheduler_str = None, None, None

            model_name_real = model_name_str if model_name_str else model_name
            sampler_name_real = sampler_name_str if sampler_name_str else sampler_name
            scheduler_real = scheduler_str if scheduler_str else scheduler

            extra_info_real = f", Extra info: {extra_info}" if extra_info else ""

            variable_map = {
                "%date": self.get_time(date_format),
                "%time": self.get_time(time_format),
                "%seed": seed,
                "%steps": steps,
                "%cfg": cfg,
                "%extension": extension,
                "%model": model_name_real,
                "%sampler": sampler_name_real,
                "%scheduler": scheduler_real,
                "%quality": jpg_webp_quality,
            }

            subfolder = self.get_path(path, variable_map)
            output_folder = Path(full_output_folder) / subfolder
            output_folder.mkdir(parents=True, exist_ok=True)
            counter = self.get_counter(output_folder)
            variable_map["%counter"] = f"{counter:05}"

            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            model_hash = (
                f"Model hash: {self.calculate_model_hash(model_name_real)}, "
                if calculate_model_hash
                else ""
            )
            comment = (
                f"{positive}\n"
                f"Negative prompt: {negative}\n"
                f"Steps: {steps}, "
                f"Sampler: {sampler_name_real}{''if scheduler_real == 'normal' else '_'+scheduler_real}, "
                f"CFG scale: {cfg}, "
                f"Seed: {seed}, "
                f"Size: {img.width if width==0 else width}x{img.height if height==0 else height}, "
                f"{model_hash}"
                f"Model: {Path(model_name_real).stem}, "
                f"Version: ComfyUI"
                f"{extra_info_real}"
            )

            stem = self.get_path(filename, variable_map)
            file = self.get_unique_filename(stem, extension, output_folder)

            if extension == "png":
                if not args.disable_metadata:
                    metadata = PngInfo()
                    metadata.add_text("parameters", comment)
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                img.save(
                    output_folder / file,
                    pnginfo=metadata,
                    compress_level=4,
                )
            else:
                img.save(
                    output_folder / file,
                    quality=jpg_webp_quality,
                    lossless=lossless_webp,
                )
                if not args.disable_metadata:
                    metadata = piexif.dump(
                        {
                            "Exif": {
                                piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                                    comment, encoding="unicode"
                                )
                            },
                        }
                    )
                    piexif.insert(metadata, str(output_folder / file))
            results.append(
                {"filename": file.name, "subfolder": str(subfolder), "type": self.type}
            )

        return {"ui": {"images": results}}

    @staticmethod
    def calculate_model_hash(model_name):
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024
        file_name = folder_paths.get_full_path("checkpoints", model_name)

        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()[:10]

    @staticmethod
    def get_counter(directory: Path):
        img_files = list(
            chain(*(directory.rglob(f"*{suffix}") for suffix in SUPPORTED_FORMATS))
        )
        return len(img_files) + 1

    @staticmethod
    def get_path(name, variable_map):
        for variable, value in variable_map.items():
            name = name.replace(variable, str(value))
        return Path(name)

    @staticmethod
    def get_time(time_format):
        now = datetime.now()
        try:
            time_str = now.strftime(time_format)
            return time_str
        except:
            return ""

    @staticmethod
    def get_unique_filename(stem: Path, extension: str, output_folder: Path):
        file = stem.with_suffix(f".{extension}")
        index = 0

        while (output_folder / file).exists():
            index += 1
            new_stem = f"{stem}_{index}"
            file = Path(new_stem).with_suffix(f".{extension}")

        return file


class SDParameterGenerator:
    ASPECT_RATIO_MAP = {
        "1:1": (512, 512),
        "4:3": (576, 448),
        "3:4": (448, 576),
        "3:2": (608, 416),
        "2:3": (416, 608),
        "16:9": (672, 384),
        "9:16": (384, 672),
        "21:9": (768, 320),
        "9:21": (320, 768),
    }

    MODEL_SCALING_FACTOR = {
        "SDv1 512px": 1.0,
        "SDv2 768px": 1.5,
        "SDXL 1024px": 2.0,
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            },
            "optional": {
                "model_version": (
                    list(SDParameterGenerator.MODEL_SCALING_FACTOR.keys()),
                    {"default": "SDv1 512px"},
                ),
                "config_name": (
                    ["none"] + folder_paths.get_filename_list("configs"),
                    {"default": "none"},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -3, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "refiner_start": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive_ascore": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
                "negative_ascore": (
                    "FLOAT",
                    {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
                "aspect_ratio": (
                    ["custom"] + list(SDParameterGenerator.ASPECT_RATIO_MAP.keys()),
                    {"default": "custom"},
                ),
                "width": (
                    "INT",
                    {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                    },
                ),
            },
        }

    RETURN_TYPES = (
        folder_paths.get_filename_list("checkpoints"),
        "MODEL",
        "CLIP",
        "VAE",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS,
        "FLOAT",
        "FLOAT",
        "INT",
        "INT",
        "INT",
        "STRING",
    )

    RETURN_NAMES = (
        "MODEL_NAME",
        "MODEL",
        "CLIP",
        "VAE",
        "SEED",
        "STEPS",
        "REFINER_START_STEP",
        "CFG",
        "SAMPLER_NAME",
        "SCHEDULER",
        "POSITIVE_ASCORE",
        "NEGATIVE_ASCORE",
        "WIDTH",
        "HEIGHT",
        "BATCH_SIZE",
        "PARAMETERS",
    )
    FUNCTION = "generate_parameter"

    CATEGORY = "SD Prompt Reader"

    def generate_parameter(
        self,
        model_version,
        ckpt_name,
        config_name,
        seed,
        steps,
        refiner_start,
        cfg,
        sampler_name,
        scheduler,
        positive_ascore,
        negative_ascore,
        aspect_ratio,
        width,
        height,
        batch_size,
        output_vae=True,
        output_clip=True,
    ):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if config_name != "none":
            config_path = folder_paths.get_full_path("configs", config_name)
            checkpoint = comfy.sd.load_checkpoint(
                config_path,
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
        else:
            checkpoint = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )[:3]

        if aspect_ratio != "custom":
            width = int(
                SDParameterGenerator.ASPECT_RATIO_MAP[aspect_ratio][0]
                * SDParameterGenerator.MODEL_SCALING_FACTOR[model_version]
            )
            height = int(
                SDParameterGenerator.ASPECT_RATIO_MAP[aspect_ratio][1]
                * SDParameterGenerator.MODEL_SCALING_FACTOR[model_version]
            )

        base_steps = int(steps * refiner_start)
        refiner_steps = steps - base_steps

        parameters = (
            f"Model: {ckpt_name},\n"
            f"Seed: {str(seed)},\n"
            f"Steps: {str(steps)},\n"
            f"CFG scale: {str(cfg)},\n"
            f"Sampler: {sampler_name},\n"
            f"Scheduler: {scheduler},\n"
            f"Size: {str(width)}x{str(height)},\n"
            f"Batch size: {str(batch_size)}\n"
        )

        return {
            "ui": {
                "text": (
                    aspect_ratio,
                    model_version,
                    width,
                    height,
                    steps,
                    refiner_start,
                    base_steps,
                    refiner_steps,
                )
            },
            "result": (
                (ckpt_name,)
                + checkpoint
                + (
                    seed,
                    steps,
                    base_steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive_ascore,
                    negative_ascore,
                    width,
                    height,
                    batch_size,
                    parameters,
                )
            ),
        }


class SDPromptMerger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_g": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
                "text_l": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_prompt"
    CATEGORY = "SD Prompt Reader"

    def merge_prompt(self, text_g, text_l):
        if text_l == "":
            return text_g
        return (text_g + "\n" + text_l,)


class SDTypeConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "model_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"forceInput": True},
                ),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"forceInput": True},
                ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )

    RETURN_NAMES = (
        "MODEL_NAME_STR",
        "SAMPLER_NAME_STR",
        "SCHEDULER_STR",
    )

    FUNCTION = "convert_string"
    CATEGORY = "SD Prompt Reader"

    def convert_string(
        self, model_name: str = "", sampler_name: str = "", scheduler: str = ""
    ):
        return (
            model_name,
            sampler_name,
            scheduler,
        )


NODE_CLASS_MAPPINGS = {
    "SDPromptReader": SDPromptReader,
    "SDPromptSaver": SDPromptSaver,
    "SDParameterGenerator": SDParameterGenerator,
    "SDPromptMerger": SDPromptMerger,
    "SDTypeConverter": SDTypeConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDPromptReader": "SD Prompt Reader",
    "SDPromptSaver": "SD Prompt Saver",
    "SDParameterGenerator": "SD Parameter Generator",
    "SDPromptMerger": "SD Prompt Merger",
    "SDTypeConverter": "SD Type Converter",
}
