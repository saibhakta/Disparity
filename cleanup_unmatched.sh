#!/bin/bash

# Usage: ./cleanup_unmatched.sh /path/to/images_dir

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/images_dir"
    exit 1
fi

images_dir="$1"
left_dir="$images_dir/left"
right_dir="$images_dir/right"

if [ ! -d "$left_dir" ] || [ ! -d "$right_dir" ]; then
    echo "Both 'left' and 'right' directories must exist inside $images_dir"
    exit 1
fi

# Create temp files
left_temp=$(mktemp)
right_temp=$(mktemp)

# Extract suffixes (e.g., 01.jpg) and sort
find "$left_dir" -type f -name "left_*.jpg" -exec basename {} \; | sed 's/^left_//' | sort > "$left_temp"
find "$right_dir" -type f -name "right_*.jpg" -exec basename {} \; | sed 's/^right_//' | sort > "$right_temp"

# Find common suffixes
common_temp=$(mktemp)
comm -12 "$left_temp" "$right_temp" > "$common_temp"

# Delete unmatched from left
while IFS= read -r f; do
    base=$(basename "$f")
    suffix="${base#left_}"
    if ! grep -qx "$suffix" "$common_temp"; then
        echo "Removing unmatched left image: $f"
        rm "$f"
    fi
done < <(find "$left_dir" -type f -name "left_*.jpg")

# Delete unmatched from right
while IFS= read -r f; do
    base=$(basename "$f")
    suffix="${base#right_}"
    if ! grep -qx "$suffix" "$common_temp"; then
        echo "Removing unmatched right image: $f"
        rm "$f"
    fi
done < <(find "$right_dir" -type f -name "right_*.jpg")

# Cleanup temp files
rm "$left_temp" "$right_temp" "$common_temp"

echo "Cleanup complete."
