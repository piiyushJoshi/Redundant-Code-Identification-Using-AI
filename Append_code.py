import json
import os

def load_dataset(filename):
    """Load existing dataset from JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def append_to_dataset(filename, new_entry):
    """
    Append a new entry to the existing dataset.
    Automatically assigns the next incremental ID.
    """
    # Load existing dataset
    dataset = load_dataset(filename)
    
    # Determine the next ID
    next_id = str(len(dataset) + 1)
    
    # Add ID to the new entry if not already present
    if 'id' not in new_entry:
        new_entry['id'] = next_id
    
    # Append the new entry
    dataset.append(new_entry)
    
    # Save the updated dataset
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"Entry added with ID {next_id}")

# Example usage
new_code_entry =  {
    "original_code": "#include <iostream>\n#include <vector>\nusing namespace std;\n\nvoid quickSort(vector<int>& arr, int low, int high) {\n    if (low < high) {\n        int pivot = partition(arr, low, high);\n        quickSort(arr, low, pivot - 1);\n        quickSort(arr, pivot + 1, high);\n    }\n}\n\nint partition(vector<int>& arr, int low, int high) {\n    int pivot = arr[high];\n    int i = (low - 1);\n\n    for (int j = low; j <= high - 1; j++) {\n        if (arr[j] < pivot) {\n            i++;\n            swap(arr[i], arr[j]);\n        }\n    }\n    swap(arr[i + 1], arr[high]);\n    return (i + 1);\n}",
    "comparison_code": "#include <iostream>\n#include <vector>\nusing namespace std;\n\nint binarySearch(vector<int>& arr, int low, int high, int key) {\n    while (low <= high) {\n        int mid = low + (high - low) / 2;\n        if (arr[mid] == key) {\n            return mid;\n        } else if (arr[mid] < key) {\n            low = mid + 1;\n        } else {\n            high = mid - 1;\n        }\n    }\n    return -1;\n}",
    "is_redundant": False,
    "language": "cpp",
    "redundancy_type": "distinct_functionality",
    "complexity_score": 5,
    "metadata": {
        "source": "manual_entry",
        "original_loc": 19,
        "comparison_loc": 15
    }
}


# File path
file_path = 'c:/Users/piyus/Desktop/Major Project/redundancy_dataset.json'

# Append the new entry
append_to_dataset(file_path, new_code_entry)