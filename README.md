# cuPQC

## Pre-requisite

To utilize cuPQC the user needs the following:

* CUDA Toolkit 12.4 or newer

* Supported CUDA compiler

  cuPQC benchmarking video: https://youtu.be/mdnXKroR1wo?si=yiTMTUy1MIzl20_7

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
nvcc -dlto -arch=native -std=c++17 -O3  -L../lib/ -lcupqc -lcuhash -o <binary_name> <file_name.cu>  -I../include/ -I../include/cupqc
```

for eg:
```shell
nvcc -dlto -arch=native -std=c++17 -O3  -L../lib/ -lcupqc  -o v4_max_bench v4_max_bench.cu  -I../include/ -I../include/cupqc
```

Make sure to adjust the `arch` as per your GPU's compute capability. Check it [here](https://developer.nvidia.com/cuda-gpus)

> Note: if programs dont run then delete the binary and build again using any of the given above two commands

### Benchmarking (these tests were done on Nvidia GH200 and files are configured according to that)

For ML-KEM 512 Benchmarking:
```shell
nvcc -dlto -arch=native -std=c++17 -O3  -L../lib/ -lcupqc  -o v4_max_bench v4_max_bench.cu  -I../include/ -I../include/cupqc
./v4_max_bench
```
For ML-KEM 512 we achieved:

* Keygen: ~20 million ops/sec
* Encapsulation: ~18.5 million ops/sec
* Decapsulation: ~18 million ops/sec


For ML-KEM 768: `./mlkem768_bench`

For ML-KEM 1024: `./mlkem1024_bench`

For ML-DSA 44: `./bench_mldsa`

For ML-DSA 65: `./mldsa65_bench`

For ML-KEM 87: `./mldsa87_bench`
