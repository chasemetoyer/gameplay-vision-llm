#!/bin/bash
# Download script for benchmark datasets
# Usage: ./scripts/download_benchmarks.sh [benchmark_name]

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

download_glitchbench() {
    echo "📦 Downloading GlitchBench..."
    mkdir -p "$DATA_DIR/glitchbench"
    cd "$DATA_DIR/glitchbench"
    
    # Download from HuggingFace
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='sail-sg/GlitchBench',
    repo_type='dataset',
    local_dir='.',
    allow_patterns=['*.parquet', '*.json']
)
"
    cd ../..
    echo "✅ GlitchBench downloaded to $DATA_DIR/glitchbench"
}

download_longvideobench() {
    echo "📦 Downloading LongVideoBench..."
    mkdir -p "$DATA_DIR/longvideobench"
    cd "$DATA_DIR/longvideobench"
    
    # Download from HuggingFace
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='longvideobench/LongVideoBench',
    repo_type='dataset',
    local_dir='.',
    allow_patterns=['*.json', '*.jsonl', 'annotations/*']
)
"
    
    # Note: Videos need to be downloaded separately from YouTube
    echo "⚠️  Video files need to be downloaded separately."
    echo "   See: https://longvideobench.github.io/"
    
    cd ../..
    echo "✅ LongVideoBench annotations downloaded to $DATA_DIR/longvideobench"
}

download_mlvu() {
    echo "📦 Downloading MLVU..."
    mkdir -p "$DATA_DIR/mlvu"
    cd "$DATA_DIR/mlvu"
    
    # MLVU from HuggingFace
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='MLVU/MLVU',
    repo_type='dataset',
    local_dir='.',
    allow_patterns=['*.json', '*.jsonl']
)
"
    cd ../..
    echo "✅ MLVU downloaded to $DATA_DIR/mlvu"
}

download_physgame() {
    echo "📦 PhysGame requires manual download."
    echo "   Visit: https://physgame.github.io/"
    echo "   Download and extract to: $DATA_DIR/physgame"
}

download_videogameqa() {
    echo "📦 VideoGameQA-Bench requires manual download."
    echo "   Visit: https://asgaardlab.github.io/videogameqa-bench/"
    echo "   Download and extract to: $DATA_DIR/videogameqa"
}

# Parse argument
BENCHMARK="${1:-all}"

case "$BENCHMARK" in
    glitchbench)
        download_glitchbench
        ;;
    longvideobench|lvb)
        download_longvideobench
        ;;
    mlvu)
        download_mlvu
        ;;
    physgame)
        download_physgame
        ;;
    videogameqa)
        download_videogameqa
        ;;
    all)
        download_glitchbench
        download_longvideobench
        download_mlvu
        download_physgame
        download_videogameqa
        ;;
    *)
        echo "Usage: $0 [glitchbench|longvideobench|mlvu|physgame|videogameqa|all]"
        exit 1
        ;;
esac

echo "🎉 Done!"
