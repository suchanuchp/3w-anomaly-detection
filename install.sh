#!/usr/bin/env bash
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0+cpu.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0+cpu.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.0+cpu.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0+cpu.html
pip install torch-geometric==1.5.0