#!/bin/bash

# convert rst to md using pandoc

INDIR=$(realpath -e "$1")
OUTDIR=$(realpath -e "$2")


for f in $(find "${INDIR}" -iname '*.rst')
do
    # extract filename without extension
    FILESTRIP=$(basename "${f%.*}")
    # extract directory of file
    FILEDIR="$(dirname "${f}")"
    # extract subdirectory of file
    # NB this takes the last directory 
    SUBDIR="$(basename "${FILEDIR}")"
    NEWDIR="${OUTDIR}"/"${SUBDIR}"
    mkdir -p "${NEWDIR}"
    echo "converting ${f} to ${NEWDIR}/$FILESTRIP.md"
    pandoc "${f}" -f rst -t markdown -o "${NEWDIR}/${FILESTRIP}.md"
    pandoc "${NEWDIR}/${FILESTRIP}.md" -o "${NEWDIR}/${FILESTRIP}.ipynb"
done


