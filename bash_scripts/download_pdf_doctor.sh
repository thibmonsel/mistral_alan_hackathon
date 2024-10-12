#!/bin/bash

# Base URL for the files
initialpath="https://www.nccn.org/professionals/physician_gls/pdf/"
current_path="$PWD"

# Array of filenames
file_list=(
    "bone.pdf"
    "colon.pdf"
    "breast.pdf"
)

# Create the pdf_data directory if it doesn't exist
mkdir -p "$current_path/pdf_data_doctor"

# Loop through the array and process each file
for filename in "${file_list[@]}"; do
    full_url="${initialpath}${filename}"
    echo "Processing: $full_url"
    
    python3 -c "
import sys
sys.path.append('$current_path')
from utils import download_pdf
download_pdf('$full_url', '$current_path/pdf_data_doctor')
"
    
done
