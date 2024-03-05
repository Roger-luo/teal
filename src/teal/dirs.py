import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_dir = os.path.relpath(os.path.join(root, "scripts"), root)
data_dir = os.path.relpath(os.path.join(root, "data"), root)
log_dir = os.path.relpath(os.path.join(root, "data", "log"), root)
