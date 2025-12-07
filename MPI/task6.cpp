#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <fstream>   


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

void exchange_halo(const Dist& d, std::vector<double>& u) {
    MPI_Status st;
    MPI_Sendrecv(&u[IDX(d, d.nx, 1)], 1, d.row_type, d.nbr_right, 0,
        &u[IDX(d, 0, 1)], 1, d.row_type, d.nbr_left, 0,
        d.comm, &st);
    MPI_Sendrecv(&u[IDX(d, 1, 1)], 1, d.row_type, d.nbr_left, 1,
        &u[IDX(d, d.nx + 1, 1)], 1, d.row_type, d.nbr_right, 1,
        d.comm, &st);
    MPI_Sendrecv(&u[IDX(d, 1, d.ny)], 1, d.col_type, d.nbr_up, 2,
        &u[IDX(d, 1, 0)], 1, d.col_type, d.nbr_down, 2,
        d.comm, &st);
    MPI_Sendrecv(&u[IDX(d, 1, 1)], 1, d.col_type, d.nbr_down, 3,
        &u[IDX(d, 1, d.ny + 1)], 1, d.col_type, d.nbr_up, 3,
        d.comm, &st);
}

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

double dotE_global(const Dist& d,
    const std::vector<double>& a,
    const std::vector<double>& b) {
    double loc = local_dotE(d, a, b);
    double g = 0.0;
    MPI_Allreduce(&loc, &g, 1, MPI_DOUBLE, MPI_SUM, d.comm);
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

Stats solve_pcg_mpi(const Dist& d,
    std::vector<double>& w,
    const std::vector<double>& F,
    int maxIter,
    double eps_rel,
    double delta) {
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
    double rr0 = dotE_global(d, r, r);
    double norm0 = std::sqrt(rr0);
    if (norm0 == 0.0) {
        st.iters = 0;
        st.resE = 0.0;
        st.dxE = 0.0;
        st.time = 0.0;
        return st;
    }
    double t0 = MPI_Wtime();
    applyDinv_local(d, r, z);
    double rz_old = dotE_global(d, r, z);
    for (int i = 1; i <= nx; ++i)
        for (int j = 1; j <= ny; ++j)
            p[IDX(d, i, j)] = z[IDX(d, i, j)];
    double norm = norm0;
    int k;
    for (k = 0; k < maxIter; ++k) {
        st.iters = k + 1;
        exchange_halo(d, p);
        applyA_local(d, p, Ap);
        double pAp = dotE_global(d, p, Ap);
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
        double rr = dotE_global(d, r, r);
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
        double dxE = std::sqrt(dotE_global(d, dw, dw));
        st.dxE = dxE;
        st.resE = norm;
        if (norm <= eps_rel * norm0 || dxE < delta) break;
        applyDinv_local(d, r, z);
        double rz = dotE_global(d, r, z);
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
    double t1 = MPI_Wtime();
    st.time = t1 - t0;
    return st;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
    if (rank == 0) {
        std::cout << "MPI PCG solver (fictitious domain), global grid "
            << M << "x" << N
            << ", processes = " << size << std::endl;
        std::cout << "Process grid Px x Py = "
            << d.Px << " x " << d.Py << std::endl;
    }
    double eps_rel = 1e-6;
    double delta = 1e-8;
    Stats st = solve_pcg_mpi(d, w, F, maxIter, eps_rel, delta);
    if (rank == 0) {
        std::cout << "[summary] iters=" << st.iters
            << ", ||r||_E=" << std::scientific << std::setprecision(6) << st.resE
            << ", ||dw||_E=" << std::fixed << std::setprecision(6) << st.dxE
            << ", time=" << std::fixed << std::setprecision(6) << st.time << " s" << std::endl;
    }
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
            int nx = allMeta[4 * rnk + 2];
            int ny = allMeta[4 * rnk + 3];
            int ny2 = ny + 2;
            const double* loc = recvbuf.data() + displs[rnk];
            for (int i = 1; i <= nx; ++i) {
                for (int j = 1; j <= ny; ++j) {
                    int gi = istart + (i - 1);
                    int gj = jstart + (j - 1);
                    if (gi < 0 || gi > d.M_global || gj < 0 || gj > d.N_global) continue;
                    int idl = i * ny2 + j;
                    globalW[gi * Ny_nodes + gj] = loc[idl];
                }
            }
        }
        std::string fname = "result_mpi_" + std::to_string(M) + "x" +
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
    MPI_Type_free(&d.row_type);
    MPI_Type_free(&d.col_type);
    MPI_Comm_free(&d.comm);
    MPI_Finalize();
    return 0;
}
