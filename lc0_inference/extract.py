import os
import shutil
import glob

SOURCE_DIR = "lc0"
DEST_DIR = "lc0_inference_standalone"

def copy_project():
    if os.path.exists(DEST_DIR):
        print(f"Removing existing {DEST_DIR}...")
        shutil.rmtree(DEST_DIR)
    
    print(f"Copying {SOURCE_DIR} to {DEST_DIR}...")
    # Ignore build folder and other artifacts
    shutil.copytree(SOURCE_DIR, DEST_DIR, ignore=shutil.ignore_patterns("build", ".git", "*.exe", "*.obj", "*.tlog"))

def clean_project():
    print("Cleaning unnecessary directories...")
    # Directories to remove
    to_remove = [
        "src/search",
        "src/selfplay",
        "src/trainingdata",
        "src/tools", # We might need some tools? No, main.cc replaces tools.
        "src/syzygy" 
    ]
    
    for relative_path in to_remove:
        path = os.path.join(DEST_DIR, relative_path)
        if os.path.exists(path):
            print(f"Removing {path}...")
            shutil.rmtree(path)
            
    # Create src/tools if it doesn't exist (or we just put main.cc in src)
    # Actually we should put main.cc in src/main.cc (overwrite lc0 main)
    
    # Overwrite main.cc
    print("Overwriting src/main.cc with inference tool...")
    shutil.copy("lc0_inference/main.cc", os.path.join(DEST_DIR, "src/main.cc"))

def patch_meson():
    print("Patching meson.build...")
    meson_path = os.path.join(DEST_DIR, "meson.build")
    with open(meson_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    skip = False
    for line in lines:
        # Remove references to deleted folders and engine files
        if any(x in line for x in ["src/search", "src/selfplay", "src/trainingdata", "src/tools", "src/syzygy", "src/engine.cc", "src/engine_loop.cc", "src/engine.h"]):
            continue
            
        if "executable('lc0'" in line:
             # Rename executable to lc0_inference
             line = line.replace("'lc0'", "'lc0_inference'")
        
        new_lines.append(line)
        
    with open(meson_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    # Also delete src/engine.cc and src/engine_loop.cc as they bring in search dependencies
    print("Removing engine files causing dependencies...")
    for f in ["src/engine.cc", "src/engine.h", "src/engine_loop.cc", "src/engine_loop.h"]:
        p = os.path.join(DEST_DIR, f)
        if os.path.exists(p):
            os.remove(p)

    # We need to remove them from meson.build files list too.
    # The simple line filter above might do it if they are listed explicitly.
    # 'src/engine.cc', is in `files += [` block.

if __name__ == "__main__":
    copy_project()
    clean_project()
    patch_meson()
    print("Extraction complete. You can now build in lc0_inference_standalone.")
