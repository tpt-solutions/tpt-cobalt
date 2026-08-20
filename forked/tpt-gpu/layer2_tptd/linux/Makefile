# SPDX-License-Identifier: Apache-2.0
#
# Makefile — TPT GPU Linux DRM Driver (Rust for Linux)
#
# Usage:
#   make KDIR=/lib/modules/$(uname -r)/build
#   sudo insmod tptd.ko
#
# Or with a full kernel source tree:
#   make KDIR=/path/to/linux-src LLVM=1

KDIR ?= /lib/modules/$(shell uname -r)/build

obj-m += tptd.o

tptd-objs := tptd_drm.o tptd_gem.o tptd_ioctls.o

# Rust source files are compiled via the kernel Kbuild Rust support.
# Each .rs file must have a corresponding .o entry here.
# The kernel Kbuild system handles the rustc invocation.

RUSTFLAGS_tptd_drm.o   := --edition=2021
RUSTFLAGS_tptd_gem.o   := --edition=2021
RUSTFLAGS_tptd_ioctls.o := --edition=2021

ccflags-y := -I$(src)/../include
rustflags-y := --edition=2021 -L$(src)/../include

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules LLVM=1

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -A

.PHONY: all clean install
