!#/bin/bash

pdflatex thesis.tex
biber thesis
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex
