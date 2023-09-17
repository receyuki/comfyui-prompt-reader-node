from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

import shutil
import folder_paths
import os

comfy_path = os.path.dirname(folder_paths.__file__)
tk_nodes_path = os.path.join(os.path.dirname(__file__))

js_dest_path = os.path.join(comfy_path, "web", "extensions", "SDPromptReader")
if not os.path.exists(js_dest_path):
    os.makedirs(js_dest_path)

js_src_path = os.path.join(tk_nodes_path, "js", "promptDisplay.js")
shutil.copy(js_src_path, js_dest_path)
