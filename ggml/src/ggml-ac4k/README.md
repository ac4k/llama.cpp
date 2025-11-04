# GGML-AC4K

## Prepare

1. cmake的版本要新，最好高于3.24
2. GGML-AC4K针对RTX5090，只支持 AC4K_CUDA_ARCHITECTURE=120（默认）
3. CUDA Toolkit Version大于等于 12.8 （Blackwell必须）

## AC4K Backend

GGML-AC4K Backend 为了不影响同时支持 GGML-CUDA，使用了独立的编译选项，包括了：

* __GGML_AC4K__
  * enable AC4K backend
* __GGML_AC4K_FORCE_MMQ__
  * use mmq kernels instead of cuBLAS
* __GGML_AC4K_FORCE_CUBLAS__
  * always use cuBLAS instead of mmq kernels
* __GGML_AC4K_PEER_MAX_BATCH_SIZE__
  * 128
* __GGML_AC4K_NO_PEER_COPY__
  * do not use peer to peer copies
* __GGML_AC4K_NO_VMM__
  * do not try to use CUDA VMM
* __GGML_AC4K_FA__
  * compile ggml FlashAttention CUDA kernels
* __GGML_AC4K_FA_ALL_QUANTS__
  * compile all quants for FlashAttention
* __GGML_AC4K_GRAPHS__
  * use CUDA graphs (llama.cpp only)
* __GGML_AC4K_GRAPHS_DEFAULT__
  * ON
* __GGML_AC4K_COMPRESSION_MODE__
  * cuda link binary compression mode
  * none;speed;balance;size(defualt)

## Build

大部分场景我们需要的是AC4K和CUDA两个backend在相同的环境下进行正确性和性能的对比：

```bash
cd llama.cpp
mkdir build
cmake -B build -DGGML_AC4K=ON -DGGML_CUDA=ON
cmake --build build --config Release  -j xx
```

也可以让AC4K和CUDA编译成不同的版本：

* GGML-CUDA使用 `CMAKE_CUDA_ARCHITECTURES` 指定ARCH，如果没有设定：
  * cmake>=3.24 and CUDAToolKit>=11.6 `CMAKE_CUDA_ARCHITECTURES`=`native`，运行时检查当前插卡类型自动选择。
  * 否则根据CUDAToolKit版本将加入 50~89 合适的版本。
* GGML-AC4K使用`AC4K_CUDA_ARCHITECTURES` 指定ARCH，当前只支持120，否则报错。

```bash
cd llama.cpp
mkdir build
cmake -B build -DGGML_AC4K=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=xx
cmake --build build --config Release  -j xx
```
