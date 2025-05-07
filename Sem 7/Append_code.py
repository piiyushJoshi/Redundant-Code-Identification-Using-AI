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
    "original_code": "int main ()\n{\n\tint a[1000],i,j,n,k;\n\tint x=0,y=0;\n\tscanf(\"%d%d\",&n,&k);\n\tfor(i=0;i<n;i++)\n\t{\n\t\tscanf(\"%d\",&a[i]);\n\t}\n\tfor(i=0;i<n;i++)\n\t{\n\t\tfor(j=i+1;j<n;j++)\n\t\t{\n\t\t\tif(a[i]+a[j]==k) \n\t\t\t{\n\t\t\t\tx=1;\n\t\t\t\ty=1;\n\t\t\t}\n\t\t\tif(x==1) break;          \n\t\t}\n\t\tif(y==1) break;\n\t}\n\tif(x==1) printf(\"yes\");\n\tif(x==0) printf(\"no\");\n\treturn 0;\n}",

    "comparison_code": "int main ()\n{\n\tint a[1000], i, j, n, k;\n\tscanf(\"%d%d\", &n, &k);\n\tfor(i = 0; i < n; i++)\n\t{\n\t\tscanf(\"%d\", &a[i]);\n\t}\n\tfor(i = 0; i < n; i++)\n\t{\n\t\tfor(j = i + 1; j < n; j++)\n\t\t{\n\t\t\tif(a[i] + a[j] == k) \n\t\t\t{\n\t\t\t\tprintf(\"yes\");\n\t\t\t\treturn 0;\n\t\t\t}\n\t\t}\n\t}\n\tprintf(\"no\");\n\treturn 0;\n}",

    "is_redundant": False,
    "language": "cpp",
    "redundancy_type": "Variable Removal",
    "complexity_score": 1,
    "metadata": {
        "source": "dataset_conversion",
        "original_loc": 19,
        "comparison_loc": 17,
        "differences": [
            "Removed unnecessary variables `x` and `y`",
            "Directly returned upon finding a valid pair instead of using flags"
        ]
    }
}




# File path
file_path = 'c:/Users/piyus/Desktop/Major Project/redundancy_dataset.json'

# Append the new entry
append_to_dataset(file_path, new_code_entry)