
import re

def get_moves_cpp(path):
    with open(path, 'r') as f:
        content = f.read()
    # Extract content between kMoveStrs = { ... };
    start = content.find('kMoveStrs[] = {')
    end = content.find('};', start)
    block = content[start:end]
    # Extract strings
    moves = re.findall(r'"([^"]+)"', block)
    return moves

def get_moves_rust(path):
    with open(path, 'r') as f:
        content = f.read()
    # Extract content between LC0_MOVES ... [ ... ]; (slice)
    # Just find strings
    moves = re.findall(r'"([a-z0-9]+)"', content)
    # Filter out potential keys if any (none in array)
    return moves

cpp_moves = get_moves_cpp(r'c:\Users\viren\Documents\GitHub\Monty0\lc0_inference_standalone\src\neural\encoder.cc')
rust_moves = get_moves_rust(r'c:\Users\viren\Documents\GitHub\Monty0\crates\datagen\src\lc0_mapping.rs')

print(f"Cpp count: {len(cpp_moves)}")
print(f"Rust count: {len(rust_moves)}")

set_cpp = set(cpp_moves)
set_rust = set(rust_moves)

diff = set_rust - set_cpp
print(f"Extra in Rust: {diff}")

diff2 = set_cpp - set_rust
print(f"Missing in Rust: {diff2}")

# Check order preservation
for i in range(min(len(cpp_moves), len(rust_moves))):
    if cpp_moves[i] != rust_moves[i]:
        print(f"Mismatch at index {i}: Cpp='{cpp_moves[i]}' Rust='{rust_moves[i]}'")
        break
