
export PYTHONPATH=$PYTHONPATH:$PWD
espeak-ng --version

python inference.py \
    --repo-id ASLP-lab/DiffRhythm2 \
    --output-dir ./results/test \
    --input-jsonl ./example/song_1.jsonl \
    --cfg-strength 3.0 \
    --max-secs 285.0 \
