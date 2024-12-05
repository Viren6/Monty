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

default:
	cargo +nightly rustc --release --bin monty --features=embed -- -C target-cpu=native --emit link=$(NAME)

montytest:
	cargo +nightly rustc --release --bin monty --features=uci-minimal,tunable -- -C target-cpu=native --emit link=$(NAME)

noembed:
	cargo +nightly rustc --release --bin monty -- -C target-cpu=native --emit link=$(NAME)

gen:
	cargo +nightly rustc --release --package datagen --bin datagen -- -C target-cpu=native --emit link=$(NAME)

release:
	cargo +nightly rustc --release --bin monty --features=embed -- --emit link=$(OLD)
	cargo +nightly rustc --release --bin monty --features=embed -- -C target-cpu=x86-64-v2 -C target-feature=+avx2 --emit link=$(AVX2)
