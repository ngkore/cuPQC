# cuPQC

## Pre-requisite

To utilize cuPQC the user needs the following:

* CUDA Toolkit 12.4 or newer

* Supported CUDA compiler

These two steps will be done using the following commands:

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

post cudakit installation steps:

```shell
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
```

```shell
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Make these changes permananet by adding them in the bashrc file

Check nvcc:
```shell
nvcc --version
```

the output should be like this:

```shell
master@master:~/slum_dawg$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```


* Supported host compiler (C++17 required)

By default present in g++ and gcc 11(default in ubuntu 22)


* (Optionally) CMake (version 3.20 or greater)

* x86_64 CPU

* A NVIDIA GPU with one the following architectures: 70, 75, 80, 86, 89, 90

In this we used `rtx a4000`. Recommended: `H100`


## Build

Firstly clone this repo

Then go to the example directory

```shell
cd cuPQC/examples/
```

Now either run the make command directly

```shell
make
```

or either run the whole command manually

```shell
nvcc -dlto -arch=native -std=c++17 -O3  -L../lib/ -lcupqc  -o <binary_name> <file_name.cu>  -I../include/ -I../include/cupqc
```

for eg:
```shell
nvcc -dlto -arch=native -std=c++17 -O3  -L../lib/ -lcupqc  -o v2_bench_refactoring v2_bench_refactoring.cu  -I../include/ -I../include/cupqc
```

> Note: if programs dont run then delete the binary and build again using any of the given above two commands

### Benchmarking

`v2_bench_refactoring.cu`

```shell
./v2_bench_refactoring
```

The output should be like this:(Current benchmarking)

```shell
./v2_bench_refactoring 
Key Generation Throughput: ~3126316.14 ops/sec
Encapsulation Throughput: ~3268814.69 ops/sec
Decapsulation Throughput: ~3134943.98 ops/sec
Benchmarking completed successfully.
```

### Key generation
`key_gen_verif_example_ml_kem.cu`

```shell
./key_gen_verif_example_ml_kem
```

### Shared secret validation
`verify_ss_encap_decap.cu`

```shell
./verify_ss_encap_decap
```
