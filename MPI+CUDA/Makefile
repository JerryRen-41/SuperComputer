# ================= 必须的变量（老师要求） =================
ARCH      ?= sm_60        # 可用 sm_35 或 sm_60，根据 GPU 选择
HOST_COMP ?= mpicxx        # MPI 主机端编译器

# ================= 编译器设置 =================
NVCC      := nvcc
CXXFLAGS  := -O3 -std=c++11
LDFLAGS   := -lm          # 必须加，否则 sqrt 链接报错

# ================= 目标程序 =================
MPI_CUDA_EXE := cuda

all: $(MPI_CUDA_EXE)

# ================= CUDA + MPI 编译规则 =================
$(MPI_CUDA_EXE): cuda.cu
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) -ccbin $(HOST_COMP) -DUSE_CUDA $< -o $@ $(LDFLAGS)

# ================= 清理 =================
clean:
	rm -f $(MPI_CUDA_EXE)
