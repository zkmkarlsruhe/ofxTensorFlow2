# build the required libs manually
# # Dan Wilcox ZKM | Hertzlab 2024

# stop on error
#.SHELLFLAGS = -e

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
        tensorflow-clean tensorflow-clobber \
        tensorflow-version

# currently macOS only
ifeq ($(PLATFORM),Darwin)
# set up build dependencies with homebrew:
# xcode-select --install
# brew install bazelisk

# bazel build options, override when building
# ex: make tensorflow-build TF_OPTS="--cpu=darwin_arm64"
TF_OPTS :=

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

# copy required files into dest $(TF_DIR)
tensorflow-copy: $(TF_DIR)/include/tensorflow/c/c_api.h $(TF_DIR)/lib/osx/libtensorflow.dylib $(TF_DIR)/LICENSE

# copy headers, this may need to be modified as TF adds or removes headers
# TODO: list required headers individually to avoid copying *everything*?
#       this is likely to break more easily as TF changes
$(TF_DIR)/include/tensorflow/c/c_api.h:
	mkdir -p $(TF_DIR)/include/tensorflow/ \
		$(TF_DIR)/include/tensorflow/c/eager $(TF_DIR)/include/tensorflow/core/platform \
		$(TF_DIR)/include/tsl/c/ $(TF_DIR)/include/tsl/platform
	cp $(BUILD_DIR)/tensorflow/tensorflow/c/*.h $(TF_DIR)/include/tensorflow/c
	cp $(BUILD_DIR)/tensorflow/tensorflow/c/eager/*.h $(TF_DIR)/include/tensorflow/c/eager
	cp $(BUILD_DIR)/tensorflow/tensorflow/core/platform/ctstring*.h $(TF_DIR)/include/tensorflow/core/platform
	cp $(BUILD_DIR)/tensorflow/third_party/xla/third_party/tsl/tsl/c/*.h $(TF_DIR)/include/tsl/c
	cp $(BUILD_DIR)/tensorflow/third_party/xla/third_party/tsl/tsl/platform/*.h $(TF_DIR)/include/tsl/platform

# copy libs
# pull version from core/public/version.h for dylib naming to replicate layout from TF build downloads
# ex: tensorflow/lib/osx should contain
# -rw-r--r--  libtensorflow.2.16.1.dylib
# lrwxr-xr-x  libtensorflow.2.dylib -> libtensorflow.2.16.1.dylib
# lrwxr-xr-x  libtensorflow.dylib -> libtensorflow.2.dylib
# -rw-r--r--  libtensorflow_framework.2.16.1.dylib
# lrwxr-xr-x  libtensorflow_framework.2.dylib -> libtensorflow_framework.2.16.1.dylib
# lrwxr-xr-x  libtensorflow_framework.dylib -> libtensorflow_framework.2.dylib
$(TF_DIR)/lib/osx/libtensorflow.dylib:
	mkdir -p $(TF_DIR)/lib/osx
	export VERSION=$(shell ./tfver.sh $(BUILD_DIR)/tensorflow) MAJOR=$(shell ./tfver.sh $(BUILD_DIR)/tensorflow major) ; \
	cp $(BUILD_DIR)/tensorflow/bazel-bin/tensorflow/libtensorflow*$${VERSION}.dylib $(TF_DIR)/lib/osx/ ; \
	cd $(TF_DIR)/lib/osx ; \
	chmod 644 libtensorflow.$${VERSION}.dylib libtensorflow_framework.$${VERSION}.dylib ; \
	ln -s libtensorflow.$${VERSION}.dylib libtensorflow.$${MAJOR}.dylib ; \
	ln -s libtensorflow.$${MAJOR}.dylib libtensorflow.dylib ; \
	ln -s libtensorflow_framework.$${VERSION}.dylib libtensorflow_framework.$${MAJOR}.dylib ; \
	ln -s libtensorflow_framework.$${MAJOR}.dylib libtensorflow_framework.dylib ;

# copy LICENSE, any resource files
$(TF_DIR)/LICENSE:
	cp $(BUILD_DIR)/tensorflow/LICENSE $(TF_DIR)/

# clean bazel build
tensorflow-clean:
	if [ -d $(BUILD_DIR)/tensorflow ] ; then cd $(BUILD_DIR)/tensorflow && bazel clean ; fi

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