#!/bin/bash

for X in `find  -type d | grep -v "\/\." | cut -d'/' -f2 | grep -v "\." | grep -v "Distributions"` ; do cat SConsTemplate | sed "s/__SF__/${X}/g" > ${X}/SConscript; done

