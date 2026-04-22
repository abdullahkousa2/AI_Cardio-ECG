#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ocr
cd "$(dirname "$0")"
python -m uvicorn app.main:app
