"""
@author: receyuki
@title: SD Prompt Reader
@nickname: SD Prompt Reader
@description: ComfyUI node version of SD Prompt Reader
"""

from .stable_diffusion_prompt_reader.sd_prompt_reader.constants import (
    SUPPORTED_FORMATS,
    MESSAGE,
)
from .stable_diffusion_prompt_reader.sd_prompt_reader.image_data_reader import (
    ImageDataReader,
)
from .stable_diffusion_prompt_reader.sd_prompt_reader.__version__ import VERSION
import folder_paths
from pathlib import Path
import os
import torch
import numpy as np
from PIL import Image, ImageOps

BLUE = "\033[1;34m"
CYAN = "\033[36m"
RESET = "\033[0m"


def output_to_terminal(text: str):
    print(f"{RESET+BLUE}" f"[SD Prompt Reader] " f"{CYAN+text+RESET}")


output_to_terminal("Reader core version: " + VERSION)


class SDPromptReader:
    def __init__(self):
        pass

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
                "data_index": (
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
        "SETTING",
    )

    FUNCTION = "load_image"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def load_image(self, image, data_index):
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

        if Path(image_path).suffix not in SUPPORTED_FORMATS:
            output_to_terminal(MESSAGE["suffix_error"][1])
            raise ValueError(MESSAGE["suffix_error"][1])

        with open(Path(image_path), "rb") as f:
            image_data = ImageDataReader(f)
            if not image_data.tool:
                output_to_terminal(MESSAGE["format_error"][1])
                raise ValueError(MESSAGE["format_error"][1])

            seed = int(
                self.param_parser(image_data.parameter.get("seed"), data_index) or 0
            )
            steps = int(
                self.param_parser(image_data.parameter.get("steps"), data_index) or 0
            )
            cfg = float(
                self.param_parser(image_data.parameter.get("cfg"), data_index) or 0
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
                image_data.setting,
            ),
        }

    @staticmethod
    def param_parser(data: str, index: int):
        data_list = data.strip("()").split(",")
        return data_list[0] if len(data_list) == 1 else data_list[index]

    @classmethod
    def IS_CHANGED(s, image, data_index):
        image_path = folder_paths.get_annotated_filepath(image)
        with open(Path(image_path), "rb") as f:
            image_data = ImageDataReader(f)
        return image_data.props

    @classmethod
    def VALIDATE_INPUTS(s, image, data_index):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"SDPromptReader": SDPromptReader}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"SDPromptReader": "SD Prompt Reader"}
