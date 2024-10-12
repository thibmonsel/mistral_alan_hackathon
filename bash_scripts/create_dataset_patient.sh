#!/bin/bash

# Set the current path and initial path for PDF files
current_path="$PWD"
initialpath="$current_path/pdf_data_patient/"

# Array of filenames
file_list=(
    "bone-patient.pdf"
    "colon-patient.pdf"
    "inflammatory-breast-patient.pdf"
)

# Ensure the pdf_data directory exists
mkdir -p "$initialpath"

# Loop through the array and create the full paths
for filename in "${file_list[@]}"; do
    full_path="${initialpath}${filename}"
    full_path_list+=("$full_path")
done

# Convert the Bash array to a comma-separated string
full_path_list_string=$(IFS=,; echo "${full_path_list[*]}")
echo "Processing files: $full_path_list_string"


    python3 -c "
from utils import create_json_file_dataset
create_json_file_dataset('$full_path_list_string', '$current_path/rag_dataset_patient.json')
"
    
done

echo "Processing complete. Output saved to $current_path/rag_dataset_patient.csv"
