import json
import os

def save_to_json(data, filename):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Write the data to a JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Creating a combined dataset with the two example JSON entries
dataset = [
    {
        "id": "1",
        "original_code": "def process_data(data_list):\n    result = []\n    for item in data_list:\n        if isinstance(item, (int, float)):\n            result.append(item * 2)\n        elif isinstance(item, str):\n            result.append(item.upper())\n        else:\n            result.append(str(item))\n    return result",
        "comparison_code": "def transform_list(input_list):\n    output = []\n    for element in input_list:\n        if type(element) in [int, float]:\n            output.append(element * 2)\n        elif type(element) == str:\n            output.append(element.upper())\n        else:\n            output.append(str(element))\n    return output",
        "is_redundant": True,
        "language": "python",
        "redundancy_type": "functional_duplicate_with_modifications",
        "complexity_score": 3,
        "metadata": {
            "source": "custom_generated",
            "original_loc": 9,
            "comparison_loc": 9,
            "differences": [
                "function_name",
                "variable_names",
                "type_checking_method"
            ]
        }
    },
    {
        "id": "2",
        "original_code": "def find_maximum(numbers):\n    max_value = numbers[0]\n    for num in numbers:\n        if num > max_value:\n            max_value = num\n    return max_value",
        "comparison_code": "def get_largest_value(values):\n    largest = values[0]\n    for val in values:\n        if val > largest:\n            largest = val\n    return largest",
        "is_redundant": True,
        "language": "python",
        "redundancy_type": "functional_duplicate_with_modifications",
        "complexity_score": 2,
        "metadata": {
            "source": "custom_generated",
            "original_loc": 6,
            "comparison_loc": 6,
            "differences": [
                "function_name",
                "variable_names"
            ]
        }
    }

]

save_to_json(dataset, 'c:\Users\piyus\Desktop\Major Project\redundancy_dataset.json')


