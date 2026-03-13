#!/usr/bin/env bash
# Download ffmpeg.wasm and mp4box.js static files for client-side video audio extraction.
# Run once after cloning: ./setup_ffmpeg.sh

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)/static/ffmpeg"
mkdir -p "$DIR"

CDN="https://cdn.jsdelivr.net/npm"

# url -> local filename (some CDN filenames differ from what we serve)
declare -A FILES=(
    ["$CDN/@ffmpeg/ffmpeg@0.12.15/dist/umd/ffmpeg.js"]="ffmpeg.js"
    ["$CDN/@ffmpeg/ffmpeg@0.12.15/dist/umd/814.ffmpeg.js"]="814.ffmpeg.js"
    ["$CDN/@ffmpeg/util@0.12.1/dist/umd/index.js"]="ffmpeg-util.js"
    ["$CDN/@ffmpeg/core@0.12.10/dist/umd/ffmpeg-core.js"]="ffmpeg-core.js"
    ["$CDN/@ffmpeg/core@0.12.10/dist/umd/ffmpeg-core.wasm"]="ffmpeg-core.wasm"
    ["$CDN/mp4box@0.5.2/dist/mp4box.all.min.js"]="mp4box.all.min.js"
)

echo "Downloading ffmpeg.wasm + mp4box.js → ${DIR}"

for url in "${!FILES[@]}"; do
    fname="${FILES[$url]}"
    if [ -f "$DIR/$fname" ]; then
        echo "  ✓ $fname (exists)"
    else
        echo "  ↓ $fname"
        curl -fsSL "$url" -o "$DIR/$fname"
    fi
done

echo "Done. $(du -sh "$DIR" | cut -f1) total."
