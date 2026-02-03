#!/bin/bash

while IFS= read -r line; do
    if [[ $line =~ ^([^:]+):([0-9]+):[0-9]+:[[:space:]]+([A-Z][0-9]+)[[:space:]] ]]; then
        filename="${BASH_REMATCH[1]}"
        linenumber="${BASH_REMATCH[2]}"
        code="${BASH_REMATCH[3]}"
        
        echo "Fixing $filename line $linenumber [$code]..."
        
        # --line-range принимает два отдельных аргумента
        autopep8 --in-place --line-range "$linenumber" "$linenumber" --select "$code" "$filename"
    fi
done < errorsflake8.txt