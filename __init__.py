from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

import shutil
import folder_paths
import os

WEB_DIRECTORY = "./js"

# remove old directory
comfy_path = os.path.dirname(folder_paths.__file__)
old_dir = os.path.join(comfy_path, "web", "extensions", "SDPromptReader")
if os.path.exists(old_dir):
    shutil.rmtree(old_dir)
