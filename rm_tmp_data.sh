find . -name ".DS_Store" -delete
find . -name "__pycache__" -type d -exec rm -r {} +
find . -name ".idea" -type d -exec rm -r {} +