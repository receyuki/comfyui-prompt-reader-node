"""
@author: receyuki
@title: SD Prompt Reader
@nickname: SD Prompt Reader
@description: The ultimate solution for managing image metadata and multi-tool compatibility. ComfyUI node version of the SD Prompt Reader
"""


import os
from datetime import datetime
from itertools import chain

import torch
import json
import re
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

ERROR_MESSAGE = {
    "format_error": "No data detected or unsupported format. "
    "Please see the README for more details.\n"
    "https://github.com/receyuki/comfyui-prompt-reader-node#supported-formats",
    "complex_workflow": "The workflow is overly complex, or unsupported custom nodes have been used. "
    "Please see the README for more details.\n"
    "https://github.com/receyuki/comfyui-prompt-reader-node#prompt-reader-node",
}


def output_to_terminal(text: str):
    print(f"{RESET+BLUE}" f"[SD Prompt Reader] " f"{CYAN+text+RESET}")


output_to_terminal("Node version: " + NODE_VERSION)
output_to_terminal("Core version: " + CORE_VERSION)


class AnyType(str):
    """A special type that can be connected to any other types. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class SDPromptReader:
    files = []
    ckpt_paths = []
    ckpt_names = []
    ckpt_stems = []

    @classmethod
    def INPUT_TYPES(s):
        for path in folder_paths.get_filename_list("checkpoints"):
            SDPromptReader.ckpt_paths.append(path)
            SDPromptReader.ckpt_names.append(Path(path).name)
            SDPromptReader.ckpt_stems.append(Path(path).stem)

        input_dir = folder_paths.get_input_directory()
        SDPromptReader.files = sorted(
            [
                f
                for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
            ]
        )
        return {
            "required": {
                "image": (SDPromptReader.files, {"image_upload": True}),
            },
            "optional": {
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
        any_type,
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
        "MODEL_NAME",
        "FILENAME",
        "SETTINGS",
    )

    FUNCTION = "load_image"
    CATEGORY = "SD Prompt Reader"
    OUTPUT_NODE = True

    def load_image(self, image, parameter_index):
        if image in SDPromptReader.files:
            image_path = folder_paths.get_annotated_filepath(image)
        elif image.startswith("pasted/"):
            image_path = folder_paths.get_annotated_filepath(image)
        else:
            image_path = image
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

        with open(file_path, "rb") as f:
            image_data = ImageDataReader(f)
            if image_data.status.name == "COMFYUI_ERROR":
                output_to_terminal(ERROR_MESSAGE["complex_workflow"])
                return self.error_output(
                    error_message=ERROR_MESSAGE["complex_workflow"],
                    image=image,
                    mask=mask,
                    width=i.width,
                    height=i.height,
                    filename=file_path.stem,
                )
            elif image_data.status.name in ["FORMAT_ERROR", "UNREAD"]:
                output_to_terminal(ERROR_MESSAGE["format_error"])
                return self.error_output(
                    error_message=ERROR_MESSAGE["format_error"],
                    image=image,
                    mask=mask,
                    width=i.width,
                    height=i.height,
                    filename=file_path.stem,
                )

            seed = int(
                self.param_parser(image_data.parameter.get("seed", 0), parameter_index)
                or 0
            )
            steps = int(
                self.param_parser(image_data.parameter.get("steps", 0), parameter_index)
                or 0
            )
            cfg = float(
                self.param_parser(image_data.parameter.get("cfg", 0), parameter_index)
                or 0
            )
            model = str(
                self.param_parser(
                    image_data.parameter.get("model", ""), parameter_index
                )
                or ""
            )
            width = int(image_data.width or 0)
            height = int(image_data.height or 0)

            output_to_terminal("Positive: \n" + image_data.positive)
            output_to_terminal("Negative: \n" + image_data.negative)
            output_to_terminal("Setting: \n" + image_data.setting)

            model = self.search_model(model)

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
                model,
                file_path.stem,
                image_data.setting,
            ),
        }

    @staticmethod
    def param_parser(data: str, index: int):
        try:
            data_list = data.strip("()").split(",")
        except AttributeError:
            return None
        else:
            return data_list[0] if len(data_list) == 1 else data_list[index]

    @staticmethod
    def search_model(model: str):
        if not model or model in SDPromptReader.ckpt_paths:
            return model

        model_path = Path(model)
        model_name = model_path.name
        model_stem = model_path.stem

        if model_name in SDPromptReader.ckpt_names:
            return SDPromptReader.ckpt_paths[
                SDPromptReader.ckpt_names.index(model_name)
            ]

        if model_stem in SDPromptReader.ckpt_stems:
            return SDPromptReader.ckpt_paths[
                SDPromptReader.ckpt_stems.index(model_stem)
            ]

        return model

    @staticmethod
    def error_output(
        error_message, image=None, mask=None, width=0, height=0, filename=""
    ):
        return {
            "ui": {"text": ("", "", error_message)},
            "result": (
                image,
                mask,
                "",
                "",
                0,
                0,
                0.0,
                width,
                height,
                "",
                filename,
                "",
            ),
        }

    @classmethod
    def IS_CHANGED(s, image, parameter_index):
        if image in SDPromptReader.files:
            image_path = folder_paths.get_annotated_filepath(image)
        else:
            image_path = image
        with open(Path(image_path), "rb") as f:
            image_data = ImageDataReader(f)
        return image_data.props

    @classmethod
    def VALIDATE_INPUTS(s, image):
        return True


class SDPromptSaver:
    model_hash_dict = {}
    vae_hash_dict = {}
    lora_hash_dict = {}
    ti_hash_dict = {}
    ti_paths = []
    ti_names = []
    ti_stems = []

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        for file in folder_paths.get_filename_list("embeddings"):
            SDPromptSaver.ti_paths.append(file)
            SDPromptSaver.ti_names.append(Path(file).name)
            SDPromptSaver.ti_stems.append(Path(file).stem)
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
                "vae_name": (folder_paths.get_filename_list("vae"),),
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
                "lora_name": any_type,
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
                "extension": (["png", "jpg", "jpeg", "webp"],),
                "calculate_hash": ("BOOLEAN", {"default": True}),
                "resource_hash": ("BOOLEAN", {"default": True}),
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
                "save_metadata_file": ("BOOLEAN", {"default": False}),
                "extra_info": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("FILENAME", "FILE_PATH", "METADATA")
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
        vae_name: str = "",
        seed: int = 0,
        steps: int = 0,
        cfg: float = 0.0,
        sampler_name: str = "",
        sampler_name_str: str = "",
        scheduler: str = "",
        scheduler_str: str = "",
        lora_name=None,
        width: int = 1,
        height: int = 1,
        positive: str = "",
        negative: str = "",
        extension: str = "png",
        calculate_hash: bool = True,
        resource_hash: bool = True,
        lossless_webp: bool = True,
        jpg_webp_quality: int = 100,
        date_format: str = "%Y-%m-%d",
        time_format: str = "%H%M%S",
        save_metadata_file: bool = False,
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

        results = []
        files = []
        comments = []
        file_paths = []
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
                "%width": width,
                "%height": height,
                "%extension": extension,
                "%model": Path(model_name_real).stem,
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

            model_hash_str = ""
            vae_hash_str = ""
            vae_str = ""
            lora_hash_dict = {}
            lora_hash_str = ""
            ti_hash_dict = {}
            ti_hash_str = ""

            if vae_name:
                vae_str = f"VAE: {Path(vae_name).stem}, "

            hashes = {}
            if calculate_hash:
                if model_name_real:
                    model_hash = self.calculate_hash(model_name_real, "model")
                    model_hash_str = f"Model hash: {model_hash}, "
                    hashes["model"] = model_hash

                if vae_name:
                    vae_hash = self.calculate_hash(vae_name, "vae")
                    vae_hash_str = f"VAE hash: {vae_hash}, "
                    hashes["vae"] = vae_hash

                if lora_name:
                    lora_names = (
                        lora_name if isinstance(lora_name, list) else [lora_name]
                    )
                    lora_names_unique = list(set(lora_names))
                    for name in lora_names_unique:
                        lora_hash = self.calculate_hash(name, "lora")
                        lora_hash_dict[Path(name).stem] = lora_hash
                        hashes[f"lora:{Path(name).stem}"] = lora_hash
                    lora_hash_items = [f"{k}: {v}" for k, v in lora_hash_dict.items()]
                    lora_hash_str_value = ", ".join(lora_hash_items)
                    lora_hash_str = f'Lora hashes: "{lora_hash_str_value}", '

                ti_pattern = (
                    r"(?:\(|\s|,)?"  # match an optional opening parenthesis, space, or comma
                    r"embedding:"  # match the literal text "embedding:"
                    r"([^\s:,()]+)"  # match a string that does not contain spaces, colons, commas, or parentheses
                    r"(?:\.(?:pt|safetensors))?"  # optionally match a file extension ".pt" or ".safetensors"
                    r"(?::\d+(?:\.\d+)?)?"  # optionally match a colon followed by numbers,
                    # with an optional decimal part (e.g., ":1" or ":1.0")
                    r"(?:\)|,|\s)?"  # optionally match a closing parenthesis, comma, or space
                )
                ti_names = re.findall(ti_pattern, f"{positive}/n{negative}")
                ti_names_with_ext = [self.search_ti(name) for name in ti_names]

                for name in ti_names_with_ext:
                    if name:
                        ti_hash = self.calculate_hash(name, "ti")
                        ti_hash_dict[Path(name).stem] = ti_hash
                        hashes[f"embed:{Path(name).stem}"] = ti_hash
                ti_hash_items = [f"{k}: {v}" for k, v in ti_hash_dict.items()]
                ti_hash_str_value = ", ".join(ti_hash_items)
                ti_hash_str = f'TI hashes: "{ti_hash_str_value}", '

            hashes_str = (
                f", Hashes: {json.dumps(hashes)}" if (hashes and resource_hash) else ""
            )

            comment = (
                f"{positive}\n"
                f"Negative prompt: {negative}\n"
                f"Steps: {steps}, "
                f"Sampler: {sampler_name_real}{''if scheduler_real == 'normal' else '_'+scheduler_real}, "
                f"CFG scale: {cfg}, "
                f"Seed: {seed}, "
                f"Size: {img.width if width==0 else width}x{img.height if height==0 else height}, "
                f"{model_hash_str}"
                f"Model: {Path(model_name_real).stem}, "
                f"{vae_hash_str}"
                f"{vae_str}"
                f"{lora_hash_str}"
                f"{ti_hash_str}"
                f"Version: ComfyUI"
                f"{hashes_str}"
                f"{extra_info_real}"
            )

            stem = self.get_path(filename, variable_map)
            file = self.get_unique_filename(stem, extension, output_folder)
            file_path = output_folder / file

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
                    file_path,
                    pnginfo=metadata,
                    compress_level=4,
                )
            else:
                img.save(
                    file_path,
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
                    piexif.insert(metadata, str(file_path))

            if save_metadata_file:
                with open(file_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
                    f.write(comment)

            results.append(
                {"filename": file.name, "subfolder": str(subfolder), "type": self.type}
            )
            files.append(str(file))
            file_paths.append(str(file_path))
            output_to_terminal("Saved file: " + str(file))
            comments.append(comment)

        return {
            "ui": {"images": results},
            "result": (
                self.unpack_singleton(files),
                self.unpack_singleton(file_paths),
                self.unpack_singleton(comments),
            ),
        }

    @staticmethod
    def calculate_hash(name, hash_type):
        match hash_type:
            case "model":
                hash_dict = SDPromptSaver.model_hash_dict
                file_name = folder_paths.get_full_path("checkpoints", name)
            case "vae":
                hash_dict = SDPromptSaver.vae_hash_dict
                file_name = folder_paths.get_full_path("vae", name)
            case "lora":
                hash_dict = SDPromptSaver.lora_hash_dict
                file_name = folder_paths.get_full_path("loras", name)
            case "ti":
                hash_dict = SDPromptSaver.ti_hash_dict
                file_name = folder_paths.get_full_path("embeddings", name)
            case _:
                return ""

        if hash_value := hash_dict.get(name):
            return hash_value

        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        hash_value = hash_sha256.hexdigest()[:10]
        hash_dict[name] = hash_value

        return hash_value

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
        file = stem.with_suffix(f"{stem.suffix}.{extension}")
        index = 0

        while (output_folder / file).exists():
            index += 1
            new_stem = Path(f"{stem}_{index}")
            file = new_stem.with_suffix(f"{new_stem.suffix}.{extension}")

        return file

    @staticmethod
    def search_ti(ti: str):
        if not ti or ti in SDPromptSaver.ti_paths:
            return ti

        if ti in SDPromptSaver.ti_stems:
            return SDPromptSaver.ti_paths[SDPromptSaver.ti_stems.index(ti)]

        if ti in SDPromptSaver.ti_names:
            return SDPromptSaver.ti_paths[SDPromptSaver.ti_names.index(ti)]

        return ""

    @staticmethod
    def unpack_singleton(arr: list):
        return arr[0] if len(arr) == 1 else arr


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

    DEFAULT_ASPECT_RATIO_DISPLAY = list(
        map(
            lambda x, scaling_factor=MODEL_SCALING_FACTOR: (
                f"{x[0]} - "
                f"{int(x[1][0]*scaling_factor['SDv1 512px'])}x"
                f"{int(x[1][1]*scaling_factor['SDv1 512px'])} | "
                f"{int(x[1][0]*scaling_factor['SDv2 768px'])}x"
                f"{int(x[1][1]*scaling_factor['SDv2 768px'])} | "
                f"{int(x[1][0]*scaling_factor['SDXL 1024px'])}x"
                f"{int(x[1][1]*scaling_factor['SDXL 1024px'])}"
            ),
            ASPECT_RATIO_MAP.items(),
        )
    )

    ckpt_list = []

    @classmethod
    def INPUT_TYPES(s):
        SDParameterGenerator.ckpt_list = folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "ckpt_name": (SDParameterGenerator.ckpt_list,),
            },
            "optional": {
                "vae_name": (
                    ["baked VAE"] + folder_paths.get_filename_list("vae"),
                    {"default": "baked VAE"},
                ),
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
                    ["custom"] + SDParameterGenerator.DEFAULT_ASPECT_RATIO_DISPLAY,
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
        folder_paths.get_filename_list("vae"),
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
        "VAE_NAME",
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
        ckpt_name,
        vae_name,
        model_version,
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

        if vae_name != "baked VAE":
            vae_name_real = vae_name
            vae_path = folder_paths.get_full_path("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)
            checkpoint = (*checkpoint[:2], vae)
            vae_str = f"VAE: {vae_name}, \n"
        else:
            vae_str = ""
            vae_name_real = ""

        if aspect_ratio != "custom":
            aspect_ratio_value = aspect_ratio.split(" - ")[0]
            width = int(
                SDParameterGenerator.ASPECT_RATIO_MAP[aspect_ratio_value][0]
                * SDParameterGenerator.MODEL_SCALING_FACTOR[model_version]
            )
            height = int(
                SDParameterGenerator.ASPECT_RATIO_MAP[aspect_ratio_value][1]
                * SDParameterGenerator.MODEL_SCALING_FACTOR[model_version]
            )

        base_steps = int(steps * refiner_start)
        refiner_steps = steps - base_steps

        if model_version == "SDXL 1024px":
            ascore = (
                f"Positive aesthetic score: {positive_ascore},\n"
                f"Negative aesthetic score: {negative_ascore},\n"
            )
        else:
            ascore = ""

        parameters = (
            f"Model: {ckpt_name},\n"
            f"{vae_str}"
            f"Seed: {str(seed)},\n"
            f"Steps: {str(steps)},\n"
            f"CFG scale: {str(cfg)},\n"
            f"Sampler: {sampler_name},\n"
            f"Scheduler: {scheduler},\n"
            f"{ascore}"
            f"Size: {str(width)}x{str(height)},\n"
            f"Batch size: {str(batch_size)}\n"
        )

        return {
            "ui": {
                "text": (
                    aspect_ratio.split(" - ")[0],
                    model_version,
                    width,
                    height,
                    steps,
                    refiner_start,
                    base_steps,
                    refiner_steps,
                    SDParameterGenerator.ASPECT_RATIO_MAP,
                    SDParameterGenerator.MODEL_SCALING_FACTOR,
                )
            },
            "result": (
                (
                    ckpt_name,
                    vae_name_real,
                )
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
            "required": {},
            "optional": {
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

    def merge_prompt(self, text_g="", text_l=""):
        return (text_g + ("\n" + text_l if text_g and text_l else text_l),)


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


class SDAnyConverter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "any_type_input": (
                    any_type,
                    {"forceInput": True},
                ),
            },
        }

    RETURN_TYPES = (any_type,)

    RETURN_NAMES = ("ANY_TYPE_OUTPUT",)

    FUNCTION = "convert_any"
    CATEGORY = "SD Prompt Reader"

    def convert_any(self, any_type_input: str = ""):
        return (any_type_input,)


class SDBatchLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": "./input/"}),
            },
            "optional": {
                "image_load_limit": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
        }

    RETURN_TYPES = (any_type,)

    RETURN_NAMES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "load_path"
    CATEGORY = "SD Prompt Reader"

    def load_path(
        self,
        path: str = "./input/",
        image_load_limit: int = 0,
        start_index: int = 0,
    ):
        if isinstance(path, list):
            files_str = [str(Path(p)) for p in path if Path(p).exists()]
            return {
                "ui": {
                    "text": ("\n".join(files_str),),
                },
                "result": (files_str,),
            }
        elif Path(path).is_file():
            return {
                "ui": {
                    "text": (str(Path(path)),),
                },
                "result": ([str(Path(path))],),
            }
        elif not Path(path).is_dir():
            raise FileNotFoundError(f"Invalid directory: {path}")

        files = list(
            filter(lambda file: file.suffix in SUPPORTED_FORMATS, Path(path).iterdir())
        )

        files = (
            sorted(files)[start_index : start_index + image_load_limit]
            if image_load_limit > 0
            else sorted(files)[start_index:]
        )

        files_str = list(map(str, files))
        return {
            "ui": {
                "text": ("\n".join(files_str),),
            },
            "result": (files_str,),
        }

    @classmethod
    def IS_CHANGED(
        s,
        path,
        image_load_limit,
        start_index,
    ):
        return os.listdir(path)


class SDParameterExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "settings": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                )
            },
            "optional": {
                "parameter": (
                    ["parameters not loaded"],
                    {"default": "parameters not loaded"},
                ),
                "value_type": (["STRING", "INT", "FLOAT"], {"default": "STRING"}),
                "parameter_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 255, "step": 1},
                ),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("VALUE",)
    OUTPUT_NODE = True
    FUNCTION = "extract_param"
    CATEGORY = "SD Prompt Reader"

    def extract_param(
        self,
        settings: str = "",
        parameter: str = "",
        value_type: str = "STRING",
        parameter_index: int = 0,
    ):
        setting_dict = self.parse_setting(settings)
        if not settings or not parameter or parameter == "parameters not loaded":
            return {
                "ui": {
                    "text": (list(setting_dict.keys()), ""),
                },
                "result": ("",),
            }

        result = setting_dict.get(parameter)

        try:
            if isinstance(result, tuple):
                result = result[parameter_index]
            if value_type == "INT":
                result = int(result)
            elif value_type == "FLOAT":
                result = float(result)
        except IndexError:
            return {
                "ui": {
                    "text": (list(setting_dict.keys()), "Parameter index out of range"),
                },
                "result": ("",),
            }
        except (ValueError, TypeError):
            return {
                "ui": {
                    "text": (
                        list(setting_dict.keys()),
                        f"{parameter}: {result}\n"
                        f"{result} is not a valid number; it will be output as STRING",
                    ),
                },
                "result": (result,),
            }
        return {
            "ui": {
                "text": (list(setting_dict.keys()), f"{parameter}: {result}"),
            },
            "result": (result,),
        }

    @staticmethod
    def parse_setting(settings):
        pattern = re.compile(r"([^:,]+):\s*\(([^)]+)\)|([^:,]+):\s*([^,]+)")

        matches = pattern.findall(settings)

        result = {}
        for match in matches:
            key, value_paren, key_nonparen, value_nonparen = match
            if key:
                key = key.strip()
                value = value_paren.strip()
                value = tuple(v.strip() for v in value.split(","))
            else:
                key = key_nonparen.strip()
                value = value_nonparen.strip()
            result[key] = value

        return result


class SDLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01},
                ),
            },
            "optional": {
                "last_lora": (any_type,),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", any_type)
    RETURN_NAMES = ("MODEL", "CLIP", "NEXT_LORA")
    FUNCTION = "load_lora"

    CATEGORY = "SD Prompt Reader"

    def load_lora(
        self, model, clip, lora_name, strength_model, strength_clip, last_lora=None
    ):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, lora_name)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )

        next_lora = last_lora + [lora_name] if last_lora else [lora_name]

        return (model_lora, clip_lora, next_lora)


class SDLoraSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            },
            "optional": {
                "last_lora": (any_type,),
            },
        }

    RETURN_TYPES = (folder_paths.get_filename_list("loras"), any_type)
    RETURN_NAMES = ("LORA_NAME", "NEXT_LORA")
    FUNCTION = "get_name"

    CATEGORY = "SD Prompt Reader"

    def get_name(self, lora_name, last_lora=None):
        next_lora = last_lora + [lora_name] if last_lora else [lora_name]
        return (lora_name, next_lora)


NODE_CLASS_MAPPINGS = {
    "SDPromptReader": SDPromptReader,
    "SDPromptSaver": SDPromptSaver,
    "SDParameterGenerator": SDParameterGenerator,
    "SDPromptMerger": SDPromptMerger,
    "SDTypeConverter": SDTypeConverter,
    "SDAnyConverter": SDAnyConverter,
    "SDBatchLoader": SDBatchLoader,
    "SDParameterExtractor": SDParameterExtractor,
    "SDLoraLoader": SDLoraLoader,
    "SDLoraSelector": SDLoraSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDPromptReader": "SD Prompt Reader",
    "SDPromptSaver": "SD Prompt Saver",
    "SDParameterGenerator": "SD Parameter Generator",
    "SDPromptMerger": "SD Prompt Merger",
    "SDTypeConverter": "SD Type Converter",
    "SDAnyConverter": "SD Any Converter",
    "SDBatchLoader": "SD Batch Loader",
    "SDParameterExtractor": "SD Parameter Extractor",
    "SDLoraLoader": "SD Lora Loader",
    "SDLoraSelector": "SD Lora Selector",
}
