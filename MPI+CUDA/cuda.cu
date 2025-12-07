#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <fstream>
#include <cstdio>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

struct Dist {
    int M_global, N_global;
    double A1, B1, A2, B2;
    double h1, h2, hmax, eps;
    MPI_Comm comm;
    int Px, Py;
    int px, py;
    int nx, ny;
    int istart, jstart;
    int nbr_left, nbr_right, nbr_down, nbr_up;
    MPI_Datatype row_type, col_type;
};

struct Stats {
    int iters;
    double resE;
    double dxE;
    double time;
};

// 计时信息
struct Timers {
    double init = 0.0;   // 初始化(分布、rhs等)
    double pcg  = 0.0;   // PCG 主循环时间
    double comm = 0.0;   // 所有MPI通信时间(halo + Allreduce等)
    double h2d  = 0.0;   // Host -> Device 拷贝时间
    double d2h  = 0.0;   // Device -> Host 拷贝时间
    double kA   = 0.0;   // kernel_applyA 时间
    double kD   = 0.0;   // kernel_applyDinv 时间
};

inline int IDX(const Dist& d, int i, int j) {
    return i * (d.ny + 2) + j;
}

void partition1D(int N, int P, int coord, int& localN, int& start) {
    int base = N / P;
    int r = N % P;
    if (coord < r) {
        localN = base + 1;
        start = coord * (base + 1);
    }
    else {
        localN = base;
        start = r * (base + 1) + (coord - r) * base;
    }
}

void choose_process_grid(int P, int Nx_nodes, int Ny_nodes, int& Px, int& Py) {
    Px = 1;
    Py = P;
    double bestScore = 1e50;
    for (int px = 1; px <= P; ++px) {
        if (P % px != 0) continue;
        int py = P / px;
        double nx = double(Nx_nodes) / px;
        double ny = double(Ny_nodes) / py;
        double ratio = nx / ny;
        if (ratio < 0.5 || ratio > 2.0) continue;
        double score = std::fabs(ratio - 1.0);
        if (score < bestScore) {
            bestScore = score;
            Px = px;
            Py = py;
        }
    }
    if (bestScore > 1e49) {
        Px = int(std::sqrt(double(P)));
        while (Px > 1 && P % Px != 0) --Px;
        Py = P / Px;
    }
}

void setup_dist(Dist& d, int M, int N, MPI_Comm comm_world) {
    d.M_global = M;
    d.N_global = N;
    d.A1 = -1.2;
    d.B1 = 1.2;
    d.A2 = -1.2;
    d.B2 = 1.2;
    d.h1 = (d.B1 - d.A1) / M;
    d.h2 = (d.B2 - d.A2) / N;
    d.hmax = std::max(d.h1, d.h2);
    d.eps = d.hmax * d.hmax;

    int size;
    MPI_Comm_size(comm_world, &size);
    int Nx_nodes = M + 1;
    int Ny_nodes = N + 1;

    choose_process_grid(size, Nx_nodes, Ny_nodes, d.Px, d.Py);

    int dims[2] = { d.Px, d.Py };
    int periods[2] = { 0, 0 };
    MPI_Cart_create(comm_world, 2, dims, periods, 0, &d.comm);

    int rank;
    MPI_Comm_rank(d.comm, &rank);
    int coords[2];
    MPI_Cart_coords(d.comm, rank, 2, coords);
    d.px = coords[0];
    d.py = coords[1];

    int nx_nodes, ny_nodes;
    partition1D(Nx_nodes, d.Px, d.px, nx_nodes, d.istart);
    partition1D(Ny_nodes, d.Py, d.py, ny_nodes, d.jstart);
    d.nx = nx_nodes;
    d.ny = ny_nodes;

    MPI_Cart_shift(d.comm, 0, 1, &d.nbr_left, &d.nbr_right);
    MPI_Cart_shift(d.comm, 1, 1, &d.nbr_down, &d.nbr_up);

    MPI_Type_contiguous(d.ny, MPI_DOUBLE, &d.row_type);
    MPI_Type_commit(&d.row_type);

    MPI_Type_vector(d.nx, 1, d.ny + 2, MPI_DOUBLE, &d.col_type);
    MPI_Type_commit(&d.col_type);
}

double overlap_len(double L0, double L1, double R0, double R1) {
    double lo = std::max(L0, R0);
    double hi = std::min(L1, R1);
    return (hi > lo) ? (hi - lo) : 0.0;
}

void build_rhs_local(const Dist& d, std::vector<double>& F) {
    int nx = d.nx, ny = d.ny;
    F.assign((nx + 2) * (ny + 2), 0.0);
    double h1 = d.h1, h2 = d.h2;
    for (int i = 1; i <= nx; ++i) {
        int gi = d.istart + (i - 1);
        for (int j = 1; j <= ny; ++j) {
            int gj = d.jstart + (j - 1);
            if (gi <= 0 || gi >= d.M_global || gj <= 0 || gj >= d.N_global) continue;
            double xL = d.A1 + (gi - 0.5) * h1;
            double xR = d.A1 + (gi + 0.5) * h1;
            double yB = d.A2 + (gj - 0.5) * h2;
            double yT = d.A2 + (gj + 0.5) * h2;

            double wx_sq = overlap_len(xL, xR, -1.0, 1.0);
            double wy_sq = overlap_len(yB, yT, -1.0, 1.0);
            double S_in_square = wx_sq * wy_sq;

            double wx_q1 = overlap_len(xL, xR, 0.0, 1.0);
            double wy_q1 = overlap_len(yB, yT, 0.0, 1.0);
            double S_in_q1 = wx_q1 * wy_q1;

            double S_D = std::max(0.0, S_in_square - S_in_q1);
            F[IDX(d, i, j)] = S_D / (h1 * h2);
        }
    }
}

// 带计时的 halo 交换
void exchange_halo(const Dist& d, std::vector<double>& u, double &t_comm) {
    MPI_Status st;
    double t0 = MPI_Wtime();

    MPI_Sendrecv(&u[IDX(d, d.nx, 1)], 1, d.row_type, d.nbr_right, 0,
                 &u[IDX(d, 0, 1)],     1, d.row_type, d.nbr_left,  0,
                 d.comm, &st);

    MPI_Sendrecv(&u[IDX(d, 1, 1)],     1, d.row_type, d.nbr_left,  1,
                 &u[IDX(d, d.nx+1, 1)],1, d.row_type, d.nbr_right, 1,
                 d.comm, &st);

    MPI_Sendrecv(&u[IDX(d, 1, d.ny)],  1, d.col_type, d.nbr_up,    2,
                 &u[IDX(d, 1, 0)],     1, d.col_type, d.nbr_down,  2,
                 d.comm, &st);

    MPI_Sendrecv(&u[IDX(d, 1, 1)],     1, d.col_type, d.nbr_down,  3,
                 &u[IDX(d, 1, d.ny+1)],1, d.col_type, d.nbr_up,    3,
                 d.comm, &st);

    t_comm += MPI_Wtime() - t0;
}

// ---------- CPU 版本算子 ----------

void applyA_local(const Dist& d,
                  const std::vector<double>& w,
                  std::vector<double>& Aw) {
    int nx = d.nx, ny = d.ny;
    Aw.assign((nx + 2) * (ny + 2), 0.0);
    double h1 = d.h1, h2 = d.h2, eps = d.eps;
    for (int i = 1; i <= nx; ++i) {
        int gi = d.istart + (i - 1);
        for (int j = 1; j <= ny; ++j) {
            int gj = d.jstart + (j - 1);
            if (gi <= 0 || gi >= d.M_global || gj <= 0 || gj >= d.N_global) continue;

            double xL = d.A1 + (gi - 0.5) * h1;
            double xR = d.A1 + (gi + 0.5) * h1;
            double yB = d.A2 + (gj - 0.5) * h2;
            double yT = d.A2 + (gj + 0.5) * h2;

            double x_vL = d.A1 + (gi - 0.5) * h1;
            double x_vR = d.A1 + (gi + 0.5) * h1;

            double ly_sq_L = overlap_len(yB, yT, -1.0, 1.0);
            double ly_q1_L = (x_vL > 0.0 && x_vL < 1.0) ? overlap_len(yB, yT, 0.0, 1.0) : 0.0;
            double lD_vL = std::max(0.0, ly_sq_L - ly_q1_L);
            double aL = (lD_vL * 1.0 + (h2 - lD_vL) * (1.0 / eps)) / h2;

            double ly_sq_R = overlap_len(yB, yT, -1.0, 1.0);
            double ly_q1_R = (x_vR > 0.0 && x_vR < 1.0) ? overlap_len(yB, yT, 0.0, 1.0) : 0.0;
            double lD_vR = std::max(0.0, ly_sq_R - ly_q1_R);
            double aR = (lD_vR * 1.0 + (h2 - lD_vR) * (1.0 / eps)) / h2;

            double y_hD = d.A2 + (gj - 0.5) * h2;
            double y_hU = d.A2 + (gj + 0.5) * h2;
            double xL_D = xL;
            double xR_D = xR;

            double lx_sq_D = overlap_len(xL_D, xR_D, -1.0, 1.0);
            double lx_q1_D = (y_hD > 0.0 && y_hD < 1.0) ? overlap_len(xL_D, xR_D, 0.0, 1.0) : 0.0;
            double lD_hD = std::max(0.0, lx_sq_D - lx_q1_D);
            double bD = (lD_hD * 1.0 + (h1 - lD_hD) * (1.0 / eps)) / h1;

            double lx_sq_U = overlap_len(xL_D, xR_D, -1.0, 1.0);
            double lx_q1_U = (y_hU > 0.0 && y_hU < 1.0) ? overlap_len(xL_D, xR_D, 0.0, 1.0) : 0.0;
            double lD_hU = std::max(0.0, lx_sq_U - lx_q1_U);
            double bU = (lD_hU * 1.0 + (h1 - lD_hU) * (1.0 / eps)) / h1;

            double wc = w[IDX(d, i, j)];
            double wl = w[IDX(d, i - 1, j)];
            double wr = w[IDX(d, i + 1, j)];
            double wd = w[IDX(d, i, j - 1)];
            double wu = w[IDX(d, i, j + 1)];

            double fxp = aR * (wr - wc);
            double fxm = aL * (wc - wl);
            double fyp = bU * (wu - wc);
            double fym = bD * (wc - wd);

            Aw[IDX(d, i, j)] = -((fxp - fxm) / (h1 * h1) + (fyp - fym) / (h2 * h2));
        }
    }
}

double local_dotE(const Dist& d,
                  const std::vector<double>& a,
                  const std::vector<double>& b) {
    int nx = d.nx, ny = d.ny;
    double s = 0.0;
    for (int i = 1; i <= nx; ++i) {
        int gi = d.istart + (i - 1);
        for (int j = 1; j <= ny; ++j) {
            int gj = d.jstart + (j - 1);
            if (gi <= 0 || gi >= d.M_global || gj <= 0 || gj >= d.N_global) continue;
            int id = IDX(d, i, j);
            s += a[id] * b[id];
        }
    }
    return s * d.h1 * d.h2;
}

// 带通信计时的全局能量内积
double dotE_global(const Dist& d,
                   const std::vector<double>& a,
                   const std::vector<double>& b,
                   double *t_comm) {
    double loc = local_dotE(d, a, b);
    double g = 0.0;
    double t0 = 0.0;
    if (t_comm) t0 = MPI_Wtime();
    MPI_Allreduce(&loc, &g, 1, MPI_DOUBLE, MPI_SUM, d.comm);
    if (t_comm) *t_comm += MPI_Wtime() - t0;
    return g;
}

void applyDinv_local(const Dist& d,
                     const std::vector<double>& r,
                     std::vector<double>& z) {
    int nx = d.nx, ny = d.ny;
    z.assign((nx + 2) * (ny + 2), 0.0);
    double h1 = d.h1, h2 = d.h2, eps = d.eps;
    for (int i = 1; i <= nx; ++i) {
        int gi = d.istart + (i - 1);
        for (int j = 1; j <= ny; ++j) {
            int gj = d.jstart + (j - 1);
            if (gi <= 0 || gi >= d.M_global || gj <= 0 || gj >= d.N_global) continue;

            double xL = d.A1 + (gi - 0.5) * h1;
            double xR = d.A1 + (gi + 0.5) * h1;
            double yB = d.A2 + (gj - 0.5) * h2;
            double yT = d.A2 + (gj + 0.5) * h2;

            double x_vL = d.A1 + (gi - 0.5) * h1;
            double x_vR = d.A1 + (gi + 0.5) * h1;

            double ly_sq_L = overlap_len(yB, yT, -1.0, 1.0);
            double ly_q1_L = (x_vL > 0.0 && x_vL < 1.0) ? overlap_len(yB, yT, 0.0, 1.0) : 0.0;
            double lD_vL = std::max(0.0, ly_sq_L - ly_q1_L);
            double aL = (lD_vL * 1.0 + (h2 - lD_vL) * (1.0 / eps)) / h2;

            double ly_sq_R = overlap_len(yB, yT, -1.0, 1.0);
            double ly_q1_R = (x_vR > 0.0 && x_vR < 1.0) ? overlap_len(yB, yT, 0.0, 1.0) : 0.0;
            double lD_vR = std::max(0.0, ly_sq_R - ly_q1_R);
            double aR = (lD_vR * 1.0 + (h2 - lD_vR) * (1.0 / eps)) / h2;

            double y_hD = d.A2 + (gj - 0.5) * h2;
            double y_hU = d.A2 + (gj + 0.5) * h2;

            double lx_sq_D = overlap_len(xL, xR, -1.0, 1.0);
            double lx_q1_D = (y_hD > 0.0 && y_hD < 1.0) ? overlap_len(xL, xR, 0.0, 1.0) : 0.0;
            double lD_hD = std::max(0.0, lx_sq_D - lx_q1_D);
            double bD = (lD_hD * 1.0 + (h1 - lD_hD) * (1.0 / eps)) / h1;

            double lx_sq_U = overlap_len(xL, xR, -1.0, 1.0);
            double lx_q1_U = (y_hU > 0.0 && y_hU < 1.0) ? overlap_len(xL, xR, 0.0, 1.0) : 0.0;
            double lD_hU = std::max(0.0, lx_sq_U - lx_q1_U);
            double bU = (lD_hU * 1.0 + (h1 - lD_hU) * (1.0 / eps)) / h1;

            double Ddiag = ((aR + aL) / (h1 * h1) + (bU + bD) / (h2 * h2));
            if (Ddiag <= 0.0) Ddiag = 1.0;

            int id = IDX(d, i, j);
            z[id] = r[id] / Ddiag;
        }
    }
}

// ---------- CUDA 部分 ----------

#ifdef USE_CUDA

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err_ = (call); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err_)); \
            MPI_Abort(MPI_COMM_WORLD, -1); \
        } \
    } while (0)

struct DistDev {
    int M_global, N_global;
    double A1, B1, A2, B2;
    double h1, h2, eps;
    int nx, ny;
    int istart, jstart;
};

__host__ DistDev makeDistDev(const Dist& d) {
    DistDev dd;
    dd.M_global = d.M_global;
    dd.N_global = d.N_global;
    dd.A1 = d.A1;
    dd.B1 = d.B1;
    dd.A2 = d.A2;
    dd.B2 = d.B2;
    dd.h1 = d.h1;
    dd.h2 = d.h2;
    dd.eps = d.eps;
    dd.nx = d.nx;
    dd.ny = d.ny;
    dd.istart = d.istart;
    dd.jstart = d.jstart;
    return dd;
}

__device__ inline int IDX_dev(const DistDev& d, int i, int j) {
    return i * (d.ny + 2) + j;
}

__device__ inline double overlap_len_dev(double L0, double L1, double R0, double R1) {
    double lo = (L0 > R0 ? L0 : R0);
    double hi = (L1 < R1 ? L1 : R1);
    return (hi > lo) ? (hi - lo) : 0.0;
}

// kernel: Aw = A(w)
__global__ void kernel_applyA(DistDev ddev, const double* w, double* Aw) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > ddev.nx || j > ddev.ny) return;

    int gi = ddev.istart + (i - 1);
    int gj = ddev.jstart + (j - 1);
    if (gi <= 0 || gi >= ddev.M_global || gj <= 0 || gj >= ddev.N_global) {
        return;
    }

    double h1 = ddev.h1;
    double h2 = ddev.h2;
    double eps = ddev.eps;

    double xL = ddev.A1 + (gi - 0.5) * h1;
    double xR = ddev.A1 + (gi + 0.5) * h1;
    double yB = ddev.A2 + (gj - 0.5) * h2;
    double yT = ddev.A2 + (gj + 0.5) * h2;

    double x_vL = ddev.A1 + (gi - 0.5) * h1;
    double x_vR = ddev.A1 + (gi + 0.5) * h1;

    double ly_sq_L = overlap_len_dev(yB, yT, -1.0, 1.0);
    double ly_q1_L = (x_vL > 0.0 && x_vL < 1.0) ? overlap_len_dev(yB, yT, 0.0, 1.0) : 0.0;
    double lD_vL = fmax(0.0, ly_sq_L - ly_q1_L);
    double aL = (lD_vL * 1.0 + (h2 - lD_vL) * (1.0 / eps)) / h2;

    double ly_sq_R = overlap_len_dev(yB, yT, -1.0, 1.0);
    double ly_q1_R = (x_vR > 0.0 && x_vR < 1.0) ? overlap_len_dev(yB, yT, 0.0, 1.0) : 0.0;
    double lD_vR = fmax(0.0, ly_sq_R - ly_q1_R);
    double aR = (lD_vR * 1.0 + (h2 - lD_vR) * (1.0 / eps)) / h2;

    double y_hD = ddev.A2 + (gj - 0.5) * h2;
    double y_hU = ddev.A2 + (gj + 0.5) * h2;

    double xL_D = xL;
    double xR_D = xR;

    double lx_sq_D = overlap_len_dev(xL_D, xR_D, -1.0, 1.0);
    double lx_q1_D = (y_hD > 0.0 && y_hD < 1.0) ? overlap_len_dev(xL_D, xR_D, 0.0, 1.0) : 0.0;
    double lD_hD = fmax(0.0, lx_sq_D - lx_q1_D);
    double bD = (lD_hD * 1.0 + (h1 - lD_hD) * (1.0 / eps)) / h1;

    double lx_sq_U = overlap_len_dev(xL_D, xR_D, -1.0, 1.0);
    double lx_q1_U = (y_hU > 0.0 && y_hU < 1.0) ? overlap_len_dev(xL_D, xR_D, 0.0, 1.0) : 0.0;
    double lD_hU = fmax(0.0, lx_sq_U - lx_q1_U);
    double bU = (lD_hU * 1.0 + (h1 - lD_hU) * (1.0 / eps)) / h1;

    int idc = IDX_dev(ddev, i, j);
    double wc = w[idc];
    double wl = w[IDX_dev(ddev, i - 1, j)];
    double wr = w[IDX_dev(ddev, i + 1, j)];
    double wd = w[IDX_dev(ddev, i, j - 1)];
    double wu = w[IDX_dev(ddev, i, j + 1)];

    double fxp = aR * (wr - wc);
    double fxm = aL * (wc - wl);
    double fyp = bU * (wu - wc);
    double fym = bD * (wc - wd);

    Aw[idc] = -((fxp - fxm) / (h1 * h1) + (fyp - fym) / (h2 * h2));
}

// kernel: z = D^{-1} r
__global__ void kernel_applyDinv(DistDev ddev, const double* r, double* z) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > ddev.nx || j > ddev.ny) return;

    int gi = ddev.istart + (i - 1);
    int gj = ddev.jstart + (j - 1);
    if (gi <= 0 || gi >= ddev.M_global || gj <= 0 || gj >= ddev.N_global) {
        return;
    }

    double h1 = ddev.h1;
    double h2 = ddev.h2;
    double eps = ddev.eps;

    double xL = ddev.A1 + (gi - 0.5) * h1;
    double xR = ddev.A1 + (gi + 0.5) * h1;
    double yB = ddev.A2 + (gj - 0.5) * h2;
    double yT = ddev.A2 + (gj + 0.5) * h2;

    double x_vL = ddev.A1 + (gi - 0.5) * h1;
    double x_vR = ddev.A1 + (gi + 0.5) * h1;

    double ly_sq_L = overlap_len_dev(yB, yT, -1.0, 1.0);
    double ly_q1_L = (x_vL > 0.0 && x_vL < 1.0) ? overlap_len_dev(yB, yT, 0.0, 1.0) : 0.0;
    double lD_vL = fmax(0.0, ly_sq_L - ly_q1_L);
    double aL = (lD_vL * 1.0 + (h2 - lD_vL) * (1.0 / eps)) / h2;

    double ly_sq_R = overlap_len_dev(yB, yT, -1.0, 1.0);
    double ly_q1_R = (x_vR > 0.0 && x_vR < 1.0) ? overlap_len_dev(yB, yT, 0.0, 1.0) : 0.0;
    double lD_vR = fmax(0.0, ly_sq_R - ly_q1_R);
    double aR = (lD_vR * 1.0 + (h2 - lD_vR) * (1.0 / eps)) / h2;

    double y_hD = ddev.A2 + (gj - 0.5) * h2;
    double y_hU = ddev.A2 + (gj + 0.5) * h2;

    double lx_sq_D = overlap_len_dev(xL, xR, -1.0, 1.0);
    double lx_q1_D = (y_hD > 0.0 && y_hD < 1.0) ? overlap_len_dev(xL, xR, 0.0, 1.0) : 0.0;
    double lD_hD = fmax(0.0, lx_sq_D - lx_q1_D);
    double bD = (lD_hD * 1.0 + (h1 - lD_hD) * (1.0 / eps)) / h1;

    double lx_sq_U = overlap_len_dev(xL, xR, -1.0, 1.0);
    double lx_q1_U = (y_hU > 0.0 && y_hU < 1.0) ? overlap_len_dev(xL, xR, 0.0, 1.0) : 0.0;
    double lD_hU = fmax(0.0, lx_sq_U - lx_q1_U);
    double bU = (lD_hU * 1.0 + (h1 - lD_hU) * (1.0 / eps)) / h1;

    double Ddiag = ((aR + aL) / (h1 * h1) + (bU + bD) / (h2 * h2));
    if (Ddiag <= 0.0) Ddiag = 1.0;

    int id = IDX_dev(ddev, i, j);
    z[id] = r[id] / Ddiag;
}

// GPU 封装：计时 H2D/D2H 和 kernel
void applyA_local_gpu(const Dist& d,
                      const std::vector<double>& w,
                      std::vector<double>& Aw,
                      DistDev ddev,
                      double* d_buf_in,
                      double* d_buf_out,
                      Timers &timers)
{
    int localSize = (d.nx + 2) * (d.ny + 2);
    size_t bytes = localSize * sizeof(double);
    Aw.assign(localSize, 0.0);

    double t0 = MPI_Wtime();
    CUDA_CHECK(cudaMemcpy(d_buf_in, w.data(), bytes, cudaMemcpyHostToDevice));
    timers.h2d += MPI_Wtime() - t0;

    dim3 block(16, 16);
    dim3 grid((d.ny + block.x - 1) / block.x,
              (d.nx + block.y - 1) / block.y);

    double t1 = MPI_Wtime();
    kernel_applyA<<<grid, block>>>(ddev, d_buf_in, d_buf_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    timers.kA += MPI_Wtime() - t1;

    double t2 = MPI_Wtime();
    CUDA_CHECK(cudaMemcpy(Aw.data(), d_buf_out, bytes, cudaMemcpyDeviceToHost));
    timers.d2h += MPI_Wtime() - t2;
}

void applyDinv_local_gpu(const Dist& d,
                         const std::vector<double>& r,
                         std::vector<double>& z,
                         DistDev ddev,
                         double* d_buf_in,
                         double* d_buf_out,
                         Timers &timers)
{
    int localSize = (d.nx + 2) * (d.ny + 2);
    size_t bytes = localSize * sizeof(double);
    z.assign(localSize, 0.0);

    double t0 = MPI_Wtime();
    CUDA_CHECK(cudaMemcpy(d_buf_in, r.data(), bytes, cudaMemcpyHostToDevice));
    timers.h2d += MPI_Wtime() - t0;

    dim3 block(16, 16);
    dim3 grid((d.ny + block.x - 1) / block.x,
              (d.nx + block.y - 1) / block.y);

    double t1 = MPI_Wtime();
    kernel_applyDinv<<<grid, block>>>(ddev, d_buf_in, d_buf_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    timers.kD += MPI_Wtime() - t1;

    double t2 = MPI_Wtime();
    CUDA_CHECK(cudaMemcpy(z.data(), d_buf_out, bytes, cudaMemcpyDeviceToHost));
    timers.d2h += MPI_Wtime() - t2;
}

#endif // USE_CUDA

// ---------- PCG 求解器 ----------

Stats solve_pcg_mpi(const Dist& d,
    std::vector<double>& w,
    const std::vector<double>& F,
    int maxIter,
    double eps_rel,
    double delta,
    Timers &timers) {

    Stats st{};
    int nx = d.nx, ny = d.ny;

    std::vector<double> r((nx + 2) * (ny + 2), 0.0);
    std::vector<double> p((nx + 2) * (ny + 2), 0.0);
    std::vector<double> z((nx + 2) * (ny + 2), 0.0);
    std::vector<double> Ap((nx + 2) * (ny + 2), 0.0);
    std::vector<double> w_prev((nx + 2) * (ny + 2), 0.0);
    std::vector<double> dw((nx + 2) * (ny + 2), 0.0);

    for (int i = 1; i <= nx; ++i)
        for (int j = 1; j <= ny; ++j)
            r[IDX(d, i, j)] = F[IDX(d, i, j)];

    double rr0 = dotE_global(d, r, r, &timers.comm);
    double norm0 = std::sqrt(rr0);
    if (norm0 == 0.0) {
        st.iters = 0;
        st.resE = 0.0;
        st.dxE = 0.0;
        st.time = 0.0;
        return st;
    }

#ifdef USE_CUDA
    DistDev ddev = makeDistDev(d);
    int localSize = (d.nx + 2) * (d.ny + 2);
    double* d_buf1 = nullptr;
    double* d_buf2 = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_buf1, localSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_buf2, localSize * sizeof(double)));
#endif

    double t_pcg0 = MPI_Wtime();

    // z = D^{-1} r
#ifdef USE_CUDA
    applyDinv_local_gpu(d, r, z, ddev, d_buf1, d_buf2, timers);
#else
    applyDinv_local(d, r, z);
#endif

    double rz_old = dotE_global(d, r, z, &timers.comm);
    for (int i = 1; i <= nx; ++i)
        for (int j = 1; j <= ny; ++j)
            p[IDX(d, i, j)] = z[IDX(d, i, j)];

    double norm = norm0;
    int k;

    for (k = 0; k < maxIter; ++k) {
        st.iters = k + 1;

        // halo for p
        exchange_halo(d, p, timers.comm);

        // Ap = A p
#ifdef USE_CUDA
        applyA_local_gpu(d, p, Ap, ddev, d_buf1, d_buf2, timers);
#else
        applyA_local(d, p, Ap);
#endif

        double pAp = dotE_global(d, p, Ap, &timers.comm);
        if (std::fabs(pAp) < 1e-30) break;
        double alpha = rz_old / pAp;

        for (int i = 1; i <= nx; ++i) {
            for (int j = 1; j <= ny; ++j) {
                int gi = d.istart + (i - 1);
                int gj = d.jstart + (j - 1);
                if (gi <= 0 || gi >= d.M_global || gj <= 0 || gj >= d.N_global) continue;
                int id = IDX(d, i, j);
                w_prev[id] = w[id];
                w[id] += alpha * p[id];
                r[id] -= alpha * Ap[id];
            }
        }

        double rr = dotE_global(d, r, r, &timers.comm);
        norm = std::sqrt(rr);

        for (int i = 1; i <= nx; ++i) {
            for (int j = 1; j <= ny; ++j) {
                int gi = d.istart + (i - 1);
                int gj = d.jstart + (j - 1);
                if (gi <= 0 || gi >= d.M_global || gj <= 0 || gj >= d.N_global) continue;
                int id = IDX(d, i, j);
                dw[id] = w[id] - w_prev[id];
            }
        }

        double dxE = std::sqrt(dotE_global(d, dw, dw, &timers.comm));
        st.dxE = dxE;
        st.resE = norm;

        if (norm <= eps_rel * norm0 || dxE < delta) break;

        // z = D^{-1} r
#ifdef USE_CUDA
        applyDinv_local_gpu(d, r, z, ddev, d_buf1, d_buf2, timers);
#else
        applyDinv_local(d, r, z);
#endif

        double rz = dotE_global(d, r, z, &timers.comm);
        double beta = rz / rz_old;
        rz_old = rz;

        for (int i = 1; i <= nx; ++i) {
            for (int j = 1; j <= ny; ++j) {
                int gi = d.istart + (i - 1);
                int gj = d.jstart + (j - 1);
                if (gi <= 0 || gi >= d.M_global || gj <= 0 || gj >= d.N_global) continue;
                int id = IDX(d, i, j);
                p[id] = z[id] + beta * p[id];
            }
        }
    }

    double t_pcg1 = MPI_Wtime();
    timers.pcg = t_pcg1 - t_pcg0;
    st.time = timers.pcg;

#ifdef USE_CUDA
    CUDA_CHECK(cudaFree(d_buf1));
    CUDA_CHECK(cudaFree(d_buf2));
#endif

    return st;
}

// ---------- main ----------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t_total0 = MPI_Wtime();
    double t_init0  = t_total0;

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -np P " << argv[0]
                      << " M N [maxIter]\n";
        }
        MPI_Finalize();
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int maxIter = (argc >= 4) ? std::atoi(argv[3]) : 8000;

    Dist d;
    setup_dist(d, M, N, MPI_COMM_WORLD);

    std::vector<double> F;
    build_rhs_local(d, F);
    std::vector<double> w((d.nx + 2) * (d.ny + 2), 0.0);

    Timers timers;
    double t_init1 = MPI_Wtime();
    timers.init = t_init1 - t_init0;

    if (rank == 0) {
        std::cout << "MPI PCG solver (fictitious domain), global grid "
                  << M << "x" << N
                  << ", processes = " << size << std::endl;
        std::cout << "Process grid Px x Py = "
                  << d.Px << " x " << d.Py << std::endl;
#ifdef USE_CUDA
        std::cout << "CUDA acceleration: ENABLED" << std::endl;
#else
        std::cout << "CUDA acceleration: DISABLED" << std::endl;
#endif
    }

    double eps_rel = 1e-6;
    double delta = 1e-8;
    Stats st = solve_pcg_mpi(d, w, F, maxIter, eps_rel, delta, timers);

    int Nx_nodes = d.M_global + 1;
    int Ny_nodes = d.N_global + 1;
    int localCount = (d.nx + 2) * (d.ny + 2);
    int meta[4] = { d.istart, d.jstart, d.nx, d.ny };
    std::vector<int> allCounts, allMeta, displs;
    std::vector<double> recvbuf;

    if (rank == 0) {
        allCounts.resize(size);
        allMeta.resize(4 * size);
    }

    MPI_Gather(&localCount, 1, MPI_INT,
               rank == 0 ? allCounts.data() : nullptr, 1, MPI_INT,
               0, d.comm);

    MPI_Gather(meta, 4, MPI_INT,
               rank == 0 ? allMeta.data() : nullptr, 4, MPI_INT,
               0, d.comm);

    if (rank == 0) {
        displs.resize(size);
        int total = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = total;
            total += allCounts[i];
        }
        recvbuf.resize(total);
    }

    MPI_Gatherv(w.data(), localCount, MPI_DOUBLE,
                rank == 0 ? recvbuf.data() : nullptr,
                rank == 0 ? allCounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, d.comm);

    if (rank == 0) {
        std::vector<double> globalW(Nx_nodes * Ny_nodes, 0.0);
        for (int rnk = 0; rnk < size; ++rnk) {
            int istart = allMeta[4 * rnk + 0];
            int jstart = allMeta[4 * rnk + 1];
            int nxr = allMeta[4 * rnk + 2];
            int nyr = allMeta[4 * rnk + 3];
            int ny2 = nyr + 2;
            const double* loc = recvbuf.data() + displs[rnk];
            for (int i = 1; i <= nxr; ++i) {
                for (int j = 1; j <= nyr; ++j) {
                    int gi = istart + (i - 1);
                    int gj = jstart + (j - 1);
                    if (gi < 0 || gi > d.M_global || gj < 0 || gj > d.N_global) continue;
                    int idl = i * ny2 + j;
                    globalW[gi * Ny_nodes + gj] = loc[idl];
                }
            }
        }

        std::string fname = "result_mpi_cuda_" + std::to_string(M) + "x" +
                            std::to_string(N) + "_t" + std::to_string(size) + ".txt";
        std::ofstream fout(fname);
        fout << std::fixed << std::setprecision(8);
        for (int gi = 0; gi <= d.M_global; ++gi) {
            for (int gj = 0; gj <= d.N_global; ++gj) {
                fout << std::setw(14) << globalW[gi * Ny_nodes + gj];
                if (gj < d.N_global) fout << " ";
            }
            fout << "\n";
        }
        fout.close();
    }

    double t_total1 = MPI_Wtime();
    double total_time = t_total1 - t_total0;
    double finalize_time = total_time - timers.init - timers.pcg;

    if (rank == 0) {
        std::cout << "[summary] iters=" << st.iters
                  << ", ||r||_E=" << std::scientific << std::setprecision(6) << st.resE
                  << ", ||dw||_E=" << std::fixed << std::setprecision(6) << st.dxE
                  << ", time=" << std::fixed << std::setprecision(6) << st.time
                  << " s" << std::endl;

        std::cout << "\n=== Timing breakdown (rank 0) ===\n";
        std::cout << "Init time          : " << timers.init   << " s\n";
        std::cout << "PCG total time     : " << timers.pcg    << " s\n";
        std::cout << "  MPI comm time    : " << timers.comm   << " s\n";
        std::cout << "  H2D copy time    : " << timers.h2d    << " s\n";
        std::cout << "  D2H copy time    : " << timers.d2h    << " s\n";
        std::cout << "  kernel A time    : " << timers.kA     << " s\n";
        std::cout << "  kernel Dinv time : " << timers.kD     << " s\n";
        std::cout << "Finalize+IO time   : " << finalize_time << " s\n";
        std::cout << "Total program time : " << total_time    << " s\n";
    }

    MPI_Type_free(&d.row_type);
    MPI_Type_free(&d.col_type);
    MPI_Comm_free(&d.comm);
    MPI_Finalize();
    return 0;
}
