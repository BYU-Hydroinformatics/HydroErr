#!/bin/bash

make html
cd ../../HydroErr_docs
cp -R html/* .
rm -rfd html
rm -rfd doctrees
