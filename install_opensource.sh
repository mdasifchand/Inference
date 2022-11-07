#!/bin/bash
set -e
# Default paths / variables
TRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/
TRT_LIB_DIR=/usr/lib/x86_64-linux-gnu/
TRT_INSTALL_DIR=/workspace/tensorrt
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
echo ${SCRIPT_DIR}
TRT_TAG=${TENSORRT_OSS_VERSION}

# getopts for the branch name
while getopts ":b:h" o; do
    case "${o}" in
    b)
        TRT_TAG=${OPTARG}
        ;;
    h)
        echo "Usage: $0 [-b <branch / tag to build>]" 1>&2; exit 0;
        ;;
    esac
done

# Clone and build OSS TensorRT based on provided tag
pushd ${SCRIPT_DIR} >/dev/null
echo "Cloning TensorRT at tag: ${TRT_TAG}"
if [ -d TensorRT ]; then rm -rf TensorRT; fi
git clone --branch ${TRT_TAG} https://github.com/NVIDIA/TensorRT.git
pushd TensorRT >/dev/null
# Copy OSS headers to default paths.
echo "Copying OSS headers to ${TRT_INCLUDE_DIR}..."
cp -r include/* ${TRT_INCLUDE_DIR}
echo "Done!"
# Sync submodules
echo "Initializing submodules..."
git submodule update --init --recursive
# If TRT_TAG is master, update the onnx-parser submodule as well.
if [[ "$TRT_TAG" == "master" ]]; then
    pushd parsers/onnx > /dev/null
    git pull origin master && git checkout master
    popd > /dev/null
fi
echo "Done!"
# Build OSS libraries
mkdir -p build && pushd build >/dev/null
echo "Running CMAKE for OSS build..."
cmake .. -DTRT_LIB_DIR=$TRT_LIB_DIR -DTRT_OUT_DIR=$PWD/out
echo "Done!"
# Make and install libraries and trtexec
echo "Running make and installing libraries and trtexec..."
make -j$(nproc) && cp -r out/libnv* $TRT_LIB_DIR && cp out/trtexec $TRT_INSTALL_DIR/bin/trtexec
echo "Done!"
popd >/dev/null
# Copy over prerequisite files to build OSS samples with cmake
echo "Copying over prerequisite files for building OSS samples to $TRT_INSTALL_DIR..."
cp CMakeLists.txt $TRT_INSTALL_DIR && cp -r cmake/ $TRT_INSTALL_DIR
# Link only if the link doesn't already exist (which can happen when script is run multiple times)
if [[ ! -L "$TRT_INSTALL_DIR/include" ]]; then
    ln -s $TRT_INCLUDE_DIR $TRT_INSTALL_DIR/include
fi
echo "Done!"
# Replace binary package samples with OSS samples and copy back python samples
echo "Replacing samples with open source versions and copying back python samples..."
if [ -d $TRT_INSTALL_DIR/samples ]; then
    mv $TRT_INSTALL_DIR/samples/python $TRT_INSTALL_DIR/python_copy
    rm -rf $TRT_INSTALL_DIR/samples
fi
cp -r samples $TRT_INSTALL_DIR
mv $TRT_INSTALL_DIR/python_copy $TRT_INSTALL_DIR/samples/python
echo "Done!"
popd >/dev/null
# Clean up
echo "Cleaning up OSS directories..."
if [ -d TensorRT ]; then rm -rf TensorRT; fi
echo "Done!"
popd >/dev/null
