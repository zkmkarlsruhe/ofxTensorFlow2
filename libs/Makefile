# build the required libs manually
# # Dan Wilcox ZKM | Hertzlab 2024

# stop on error
.SHELLFLAGS = -e

# local temp build dir
BUILD_DIR = build

# Windows, Linux, Darwin (macOS), etc
PLATFORM = $(shell uname 2>/dev/null || echo Unknown)

.PHONY: clean clobber

all: tensorflow

# remove build
clean: tensorflow-clean
	rm -rf $(BUILD_DIR)

clobber: tensorflow-clobber

##### tensorflow

# git tag, branch, or commit
TF_VER := v2.16.1

# git repo url
TF_URL = https://github.com/tensorflow/tensorflow.git

# output dir
TF_DIR = tensorflow

.PHONY: tensorflow tensorflow-download tensorflow-build tensorflow-copy \
        tensorflow-clean tensorflow-clobber

# currently macOS only
ifeq ($(PLATFORM),Darwin)
# set up build dependencies with homebrew:
# xcode-select --install
# brew install bazelisk

# bazel build options, override when building: make tensorflow-build TD_OPTS="--cpu=darwin_arm64 --host_cpu=darwin_x86_64"
TF_OPTS := --cpu=darwin_arm64 --host_cpu=darwin_x86_64

tensorflow: tensorflow-download tensorflow-build tensorflow-copy tensorflow-clean

tensorflow-download: $(BUILD_DIR)/tensorflow

# download via shallow clone
$(BUILD_DIR)/tensorflow:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && \
	git clone --branch $(TF_VER) --depth 1 $(TF_URL)

# build, requires answering configure questions
tensorflow-build:
	cd $(BUILD_DIR)/tensorflow && \
	./configure && \
	bazel build --config opt $(TF_OPTS) //tensorflow/tools/lib_package:libtensorflow

tensorflow-copy: $(TF_DIR)/lib/osx/libtensorflow.dylib

# copy headers and libs
$(TF_DIR)/lib/osx/libtensorflow.dylib:
	mkdir -p $(TF_DIR)/include/tensorflow/ $(TF_DIR)/lib/osx
	mkdir -p $(TF_DIR)/include/tensorflow/c/eager $(TF_DIR)/include/tensorflow/core/platform
	mkdir -p $(TF_DIR)/include/tsl/c/ $(TF_DIR)/include/tsl/platform
	cp $(BUILD_DIR)/tensorflow/tensorflow/c/*.h $(TF_DIR)/include/tensorflow/c
	cp $(BUILD_DIR)/tensorflow/tensorflow/c/eager/*.h $(TF_DIR)/include/tensorflow/c/eager
	cp $(BUILD_DIR)/tensorflow/tensorflow/core/platform/ctstring*.h $(TF_DIR)/include/tensorflow/core/platform
	cp $(BUILD_DIR)/tensorflow/third_party/xla/third_party/tsl/tsl/c/*.h $(TF_DIR)/include/tsl/c
	cp $(BUILD_DIR)/tensorflow/third_party/xla/third_party/tsl/tsl/platform/*.h $(TF_DIR)/include/tsl/platform
	cp $(BUILD_DIR)/tensorflow/bazel-bin/tensorflow/*.dylib $(TF_DIR)/lib/osx/
	cp $(BUILD_DIR)/tensorflow/LICENSE $(TF_DIR)/
	cd $(TF_DIR)/lib/osx && ln -s libtensorflow.$(TF_VER).dylib libtensorflow.2.dylib
	cd $(TF_DIR)/lib/osx && ln -s libtensorflow.2.dylib libtensorflow.dylib
	cd $(TF_DIR)/lib/osx && ln -s libtensorflow_framework.2.dylib libtensorflow_framework.dylib

# clean bazel build
tensorflow-clean:
	cd $(BUILD_DIR)/tensorflow && bazel clean

# remove output dir
tensorflow-clobber:
	rm -rf $(TF_DIR)/*

else
# non-macOS dummies

tensorflow:
	echo "building libtensorflow from source currently only supported on macOS"

tensorflow-download:

tensorflow-build:

tensorflow-copy:

tensorflow-clean:

tensorflow-clobber:

endif