from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

import shutil
import folder_paths
import os

comfy_path = os.path.dirname(folder_paths.__file__)
tk_nodes_path = os.path.join(os.path.dirname(__file__))

js_dest_path = os.path.join(comfy_path, "web", "extensions", "SDPromptReader")
os.makedirs(js_dest_path, exist_ok=True)

files_to_copy = ["promptDisplay.js", "parameterDisplay.js", "seedGen.js"]

for file in files_to_copy:
    js_src_path = os.path.join(tk_nodes_path, "js", file)
    shutil.copy(js_src_path, js_dest_path)
