# Minimal build for the C ABI example on Windows (clang + MSVC libs).
# Adjust VS/SDK paths if your toolchain layout differs.
VS ?= C:\Program Files\Microsoft Visual Studio\2022\Community
VCTOOLS := $(VS)\VC\Tools\MSVC
VCVER := $(lastword $(wildcard $(VCTOOLS)\*))
SDKINC := $(lastword $(wildcard "C:\Program Files (x86)\Windows Kits\10\Include\*))
SDKLIB := $(lastword $(wildcard "C:\Program Files (x86)\Windows Kits\10\Lib\*"))
RUST_TARGET ?= debug

INCLUDES := -I include -I "$(VCVER)\include" -I "$(SDKINC)\shared" -I "$(SDKINC)\um" -I "$(SDKINC)\ucrt"
LIBPATHS := -L ../../target/$(RUST_TARGET) -L "$(VCVER)\lib\x64" -L "$(SDKLIB)\um\x64" -L "$(SDKLIB)\ucrt\x64"
SYSLIBS := -lntdll -lws2_32 -ladvapi32 -luserenv -lbcrypt -lkernel32 -lsynchronization

.PHONY: build-runtime example clean

build-runtime:
	cd ../.. && cargo build -p out-gpu-runtime-c

example: build-runtime
	clang examples/c/hello_device.c $(INCLUDES) $(LIBPATHS) -ltptr_c $(SYSLIBS) -o examples/c/hello_device.exe
	copy ..\..\target\$(RUST_TARGET)\tptr_c.dll examples\c\ /Y
	examples\c\hello_device.exe

clean:
	del /Q examples\c\hello_device.exe examples\c\tptr_c.dll 2>nul || exit 0
