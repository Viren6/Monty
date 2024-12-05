SHELL := /bin/bash

EXE = monty

ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
	OLD := monty-$(VER).exe
	AVX2 := monty-$(VER)-avx2.exe
else
	NAME := $(EXE)
	OLD := monty-$(VER)
	AVX2 := monty-$(VER)-avx2
endif

# Ensure nightly is installed
.PHONY: install-nightly
install-nightly:
	source $$HOME/.cargo/env && rustup install nightly

montytest: install-nightly
	cargo +nightly rustc --release --bin monty --features=uci-minimal,tunable -- -C target-cpu=native --emit link=$(NAME)
