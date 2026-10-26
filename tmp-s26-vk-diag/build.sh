#!/usr/bin/env bash
# TEMP (s26 Vulkan diagnosis): builds the diagnostic tools for Android arm64 from a fabric checkout
# with ggml-vk-diag.patch applied, and stages them with run-diag.sh and its inputs.
# Usage: build.sh <fabric source dir> <output dir>. Needs ANDROID_NDK and VULKAN_SDK (for glslc).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$1"
OUT="$2"
: "${ANDROID_NDK:?}" "${VULKAN_SDK:?}"

git -C "$SRC" apply "$HERE/ggml-vk-diag.patch"

cmake -S "$SRC" -B "$SRC/build-android" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_VULKAN=ON -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
    -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
    -DVulkan_GLSLC_EXECUTABLE="$VULKAN_SDK/bin/glslc"
cmake --build "$SRC/build-android" --target test-backend-ops test-export-graph-ops llama-eval-callback llama-finetune-lora

TOOLS="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64"
B="$SRC/build-android/bin"
mkdir -p "$OUT/bin" "$OUT/lib"
cp "$B/test-backend-ops" "$B/test-export-graph-ops" "$B/llama-eval-callback" "$B/llama-finetune-lora" "$OUT/bin/"
cp "$B"/libggml*.so "$B/libllama.so" "$TOOLS/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "$OUT/lib/"
"$TOOLS/bin/llvm-strip" --strip-unneeded "$OUT"/bin/* "$OUT"/lib/*.so
cp "$HERE/run-diag.sh" "$HERE/prompt.txt" "$HERE/train.jsonl" "$OUT/"
"$VULKAN_SDK/bin/glslc" --version | head -1 > "$OUT/build-info.txt"
git -C "$SRC" log -1 --format='fabric %H %s' >> "$OUT/build-info.txt"
du -sh "$OUT"
