#!/bin/bash
# Script to rename screenshot files in chronological order

counter=1
while IFS= read -r file; do
    # Format counter with leading zeros (001, 002, etc.)
    num=$(printf "%03d" $counter)
    
    # Create new filename
    newname="fig-${num}.png"
    
    # Rename the file
    mv "$file" "$newname"
    echo "Renamed: $file -> $newname"
    
    ((counter++))
done < filelist.txt

echo "Total files renamed: $((counter-1))"
