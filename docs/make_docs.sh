#!/bin/bash

make html
cd ../../HydroErr_docs
cp -R html/* .
rm -rfd html
rm -rfd doctrees

git add *
git commit -m "Update Docs"
git push

