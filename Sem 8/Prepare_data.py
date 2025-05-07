import os
import json
import random
import itertools

# Define dataset path and output JSON file
DATASET_PATH = "C:/Users/piyus/Desktop/Major Project/ProgramData"  
OUTPUT_JSON = "Code_Similarity_Dataset.json"
PAIRS_PER_FOLDER = 500  # Limit to 500 positive and 500 negative pairs per folder

# Function to read code files from a folder 
def read_code_files(folder_path):
    code_snippets = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_snippets.append(f.read().strip())
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="ISO-8859-1") as f:
                        code_snippets.append(f.read().strip())
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="Windows-1252", errors="replace") as f:
                        code_snippets.append(f.read().strip())  # Replaces unrecognized characters
    return code_snippets

# Function to generate limited pairs
def generate_limited_pairs():
    all_folders = sorted(os.listdir(DATASET_PATH))
    folder_codes = {}

    # Read and store a reference of each folder
    for folder in all_folders:
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.isdir(folder_path):
            folder_codes[folder] = read_code_files(folder_path)

    # Open output file in append mode
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f_out:
        f_out.write("[\n")  # Start JSON array
        first_entry = True

        # Generate positive and negative pairs
        for folder in all_folders:
            folder_path = os.path.join(DATASET_PATH, folder)
            if not os.path.isdir(folder_path):
                continue

            codes = folder_codes[folder]
            if len(codes) < 2:
                continue  # Skip folders with insufficient samples

            # Generate at most 500 positive pairs
            pos_pairs = list(itertools.combinations(codes, 2))
            random.shuffle(pos_pairs)  # Shuffle to get diverse pairs
            pos_pairs = pos_pairs[:PAIRS_PER_FOLDER]  # Limit to 500

            for code1, code2 in pos_pairs:
                if not first_entry:
                    f_out.write(",\n")
                json.dump({"code1": code1, "code2": code2, "label": 1}, f_out)
                first_entry = False

            # Generate exactly 500 negative pairs per folder
            other_folders = [f for f in all_folders if f != folder]
            neg_pairs = []
            while len(neg_pairs) < PAIRS_PER_FOLDER:
                f1, f2 = random.sample(other_folders, 2)
                if folder_codes[f1] and folder_codes[f2]:  # Ensure both have codes
                    code1 = random.choice(folder_codes[f1])
                    code2 = random.choice(folder_codes[f2])
                    neg_pairs.append((code1, code2))

            for code1, code2 in neg_pairs:
                if not first_entry:
                    f_out.write(",\n")
                json.dump({"code1": code1, "code2": code2, "label": 0}, f_out)
                first_entry = False

        f_out.write("\n]")  # Close JSON array

    print(f"Optimized dataset saved as {OUTPUT_JSON}")

# Run the function
generate_limited_pairs()
