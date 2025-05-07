import json
import random

def generate_code_pair():
    # This is a simplified example. In practice, you'd want more complex and varied code generation.
    functions = [
        "def calculate_area(l, w):\n    return l * w",
        "def compute_rectangle_area(length, width):\n    area = length * width\n    return area",
        "def area_of_rectangle(x, y):\n    return x * y"
    ]
    return random.choice(functions), random.choice(functions)

def generate_dataset(num_entries, output_file):
    with open(output_file, 'w') as f:
        for i in range(num_entries):
            original, comparison = generate_code_pair()
            entry = {
                "id": str(i),
                "original_code": original,
                "comparison_code": comparison,
                "is_redundant": original != comparison,  # Simplified redundancy check
                "language": "python",
                "redundancy_type": "functional_duplicate" if original != comparison else "exact_duplicate",
                "complexity_score": random.randint(1, 5)
            }
            json.dump(entry, f)
            f.write('\n')

# Usage
generate_dataset(1000, 'dataset.jsonl')