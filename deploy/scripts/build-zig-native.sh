#!/bin/bash
# Build Zig native SIMD library for Ananke
#
# Usage:
#   ./build-zig-native.sh [--release|--debug] [--install]
#
# Requirements:
#   - Zig 0.13.0+ (https://ziglang.org/download/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ZIG_DIR="$REPO_ROOT/python/sglang/srt/constrained/ananke/zig"

# Defaults
BUILD_MODE="ReleaseFast"
INSTALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            BUILD_MODE="ReleaseFast"
            shift
            ;;
        --debug)
            BUILD_MODE="Debug"
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--release|--debug] [--install]"
            echo ""
            echo "Options:"
            echo "  --release   Build with ReleaseFast optimization (default)"
            echo "  --debug     Build with debug symbols"
            echo "  --install   Install library to system location"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for Zig
if ! command -v zig &> /dev/null; then
    echo "Error: Zig compiler not found"
    echo ""
    echo "Install Zig:"
    echo "  macOS:   brew install zig"
    echo "  Linux:   See https://ziglang.org/download/"
    echo "  Docker:  Use --build-arg BUILD_ZIG_NATIVE=1"
    exit 1
fi

ZIG_VERSION=$(zig version)
echo "Using Zig $ZIG_VERSION"

# Check if Zig directory exists
if [[ ! -d "$ZIG_DIR" ]]; then
    echo "Error: Zig source directory not found at $ZIG_DIR"
    echo "Make sure you're running from the sglang repository"
    exit 1
fi

# Build
echo "Building Ananke Zig native library..."
echo "  Mode: $BUILD_MODE"
echo "  Directory: $ZIG_DIR"

cd "$ZIG_DIR"

# Detect CPU features for optimal build
ARCH=$(uname -m)
case $ARCH in
    x86_64)
        # Check for AVX2 support
        if grep -q avx2 /proc/cpuinfo 2>/dev/null || sysctl -n machdep.cpu.features 2>/dev/null | grep -q AVX2; then
            echo "  CPU Features: AVX2 detected"
        fi
        ;;
    aarch64|arm64)
        echo "  CPU Features: ARM NEON"
        ;;
esac

# Build with Zig
if [[ "$BUILD_MODE" == "Debug" ]]; then
    zig build
else
    zig build -Doptimize=ReleaseFast
fi

# Check output
if [[ -f "zig-out/lib/libananke_simd.so" ]] || [[ -f "zig-out/lib/libananke_simd.dylib" ]]; then
    echo ""
    echo "Build successful!"
    ls -lh zig-out/lib/libananke_simd.*
else
    echo "Error: Library not found after build"
    exit 1
fi

# Install if requested
if [[ "$INSTALL" == true ]]; then
    INSTALL_DIR="/usr/local/lib"
    echo ""
    echo "Installing to $INSTALL_DIR..."

    if [[ -f "zig-out/lib/libananke_simd.so" ]]; then
        sudo cp zig-out/lib/libananke_simd.so "$INSTALL_DIR/"
        sudo ldconfig 2>/dev/null || true
    elif [[ -f "zig-out/lib/libananke_simd.dylib" ]]; then
        sudo cp zig-out/lib/libananke_simd.dylib "$INSTALL_DIR/"
    fi

    echo "Installation complete"
fi

echo ""
echo "To use the native library, ensure it's in your library path:"
echo "  export LD_LIBRARY_PATH=\"$ZIG_DIR/zig-out/lib:\$LD_LIBRARY_PATH\""
