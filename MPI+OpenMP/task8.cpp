#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <vector>
#include <iostream>
#include <cstdlib>

struct Dist {
    int M, N;
    double A1, B1, A2, B2, h1, h2;
    MPI_Comm comm;
    int Px, Py;
    int px, py;
    int nx, ny;
    int istart, jstart;
    int nbr_left, nbr_right, nbr_down, nbr_up;
    MPI_Datatype row_type, col_type;
    std::vector<char> mask;
};

inline int IDX(const Dist& d, int i, int j) {
    return i * (d.ny + 2) + j;
}

inline bool inDomain(double x, double y) {
    if (x <= -1.0 || x >= 1.0 || y <= -1.0 || y >= 1.0) return false;
    if (x > 0.0 && y > 0.0) return false;
    return true;
}

void choose_PxPy(int P, int M, int N, int& Px, int& Py) {
    Px = 1;
    Py = P;
    double best = 1e18;
    for (int x = 1; x <= P; x++) {
        if (P % x) continue;
        int y = P / x;
        double diff = std::fabs(double(M) / x - double(N) / y);
        if (diff < best) {
            best = diff;
            Px = x;
            Py = y;
        }
    }
}

void partition(int N, int P, int r, int& localN, int& start) {
    int base = N / P, rem = N % P;
    if (r < rem) {
        localN = base + 1;
        start = r * (base + 1);
    }
    else {
        localN = base;
        start = rem * (base + 1) + (r - rem) * base;
    }
}

void setup_dist(Dist& d, int M, int N, MPI_Comm world) {
    d.M = M;
    d.N = N;
    d.A1 = -1.2;
    d.B1 = 1.2;
    d.A2 = -1.2;
    d.B2 = 1.2;
    d.h1 = (d.B1 - d.A1) / M;
    d.h2 = (d.B2 - d.A2) / N;

    int P;
    MPI_Comm_size(world, &P);
    choose_PxPy(P, M + 1, N + 1, d.Px, d.Py);

    int dims[2] = { d.Px, d.Py };
    int periods[2] = { 0, 0 };
    MPI_Cart_create(world, 2, dims, periods, 0, &d.comm);

    int rank;
    MPI_Comm_rank(d.comm, &rank);
    int coords[2];
    MPI_Cart_coords(d.comm, rank, 2, coords);
    d.px = coords[0];
    d.py = coords[1];

    partition(M + 1, d.Px, d.px, d.nx, d.istart);
    partition(N + 1, d.Py, d.py, d.ny, d.jstart);

    MPI_Cart_shift(d.comm, 0, 1, &d.nbr_left, &d.nbr_right);
    MPI_Cart_shift(d.comm, 1, 1, &d.nbr_down, &d.nbr_up);

    MPI_Type_contiguous(d.ny, MPI_DOUBLE, &d.row_type);
    MPI_Type_commit(&d.row_type);

    MPI_Type_vector(d.nx, 1, d.ny + 2, MPI_DOUBLE, &d.col_type);
    MPI_Type_commit(&d.col_type);
}

void halo_exchange(const Dist& d, std::vector<double>& f) {
    MPI_Request reqs[8];
    int r = 0;

    if (d.nbr_left != MPI_PROC_NULL) {
        MPI_Irecv(&f[IDX(d, 0, 1)], 1, d.row_type, d.nbr_left, 10, d.comm, &reqs[r++]);
        MPI_Isend(&f[IDX(d, 1, 1)], 1, d.row_type, d.nbr_left, 11, d.comm, &reqs[r++]);
    }
    if (d.nbr_right != MPI_PROC_NULL) {
        MPI_Irecv(&f[IDX(d, d.nx + 1, 1)], 1, d.row_type, d.nbr_right, 11, d.comm, &reqs[r++]);
        MPI_Isend(&f[IDX(d, d.nx, 1)], 1, d.row_type, d.nbr_right, 10, d.comm, &reqs[r++]);
    }
    if (d.nbr_down != MPI_PROC_NULL) {
        MPI_Irecv(&f[IDX(d, 1, 0)], 1, d.col_type, d.nbr_down, 20, d.comm, &reqs[r++]);
        MPI_Isend(&f[IDX(d, 1, 1)], 1, d.col_type, d.nbr_down, 21, d.comm, &reqs[r++]);
    }
    if (d.nbr_up != MPI_PROC_NULL) {
        MPI_Irecv(&f[IDX(d, 1, d.ny + 1)], 1, d.col_type, d.nbr_up, 21, d.comm, &reqs[r++]);
        MPI_Isend(&f[IDX(d, 1, d.ny)], 1, d.col_type, d.nbr_up, 20, d.comm, &reqs[r++]);
    }

    if (r > 0) MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);
}

void applyA(const Dist& d,
    const std::vector<double>& u,
    std::vector<double>& Au)
{
    Au.assign((d.nx + 2) * (d.ny + 2), 0.0);

#pragma omp parallel for collapse(2)
    for (int i = 1; i <= d.nx; i++)
        for (int j = 1; j <= d.ny; j++) {
            int id = IDX(d, i, j);
            if (!d.mask[id]) continue;
            Au[id] =
                -u[IDX(d, i - 1, j)]
                - u[IDX(d, i + 1, j)]
                - u[IDX(d, i, j - 1)]
                - u[IDX(d, i, j + 1)]
                + 4.0 * u[id];
        }
}

void applyDinv(const Dist& d,
    const std::vector<double>& r,
    std::vector<double>& z)
{
    z.assign((d.nx + 2) * (d.ny + 2), 0.0);

#pragma omp parallel for collapse(2)
    for (int i = 1; i <= d.nx; i++)
        for (int j = 1; j <= d.ny; j++) {
            int id = IDX(d, i, j);
            if (!d.mask[id]) continue;
            z[id] = 0.25 * r[id];
        }
}

double dot_local(const Dist& d,
    const std::vector<double>& x,
    const std::vector<double>& y)
{
    double s = 0.0;
#pragma omp parallel for collapse(2) reduction(+:s)
    for (int i = 1; i <= d.nx; i++)
        for (int j = 1; j <= d.ny; j++) {
            int id = IDX(d, i, j);
            if (!d.mask[id]) continue;
            s += x[id] * y[id];
        }
    return s;
}

struct Stats {
    int iters;
    double rnorm;
    double time;
};

Stats solve_pcg(const Dist& d,
    std::vector<double>& u,
    const std::vector<double>& f)
{
    int sz = (d.nx + 2) * (d.ny + 2);
    std::vector<double> r(sz), z(sz), p(sz), Ap(sz);

    double t0 = MPI_Wtime();

    halo_exchange(d, u);
    applyA(d, u, Ap);

#pragma omp parallel for collapse(2)
    for (int i = 1; i <= d.nx; i++)
        for (int j = 1; j <= d.ny; j++) {
            int id = IDX(d, i, j);
            r[id] = d.mask[id] ? (f[id] - Ap[id]) : 0.0;
        }

    applyDinv(d, r, z);

    double rz = dot_local(d, r, z);
    MPI_Allreduce(MPI_IN_PLACE, &rz, 1, MPI_DOUBLE, MPI_SUM, d.comm);

    p = z;

    int k;
    for (k = 0; k < 100000; k++) {
        halo_exchange(d, p);
        applyA(d, p, Ap);

        double pAp = dot_local(d, p, Ap);
        MPI_Allreduce(MPI_IN_PLACE, &pAp, 1, MPI_DOUBLE, MPI_SUM, d.comm);

        double alpha = rz / pAp;

#pragma omp parallel for collapse(2)
        for (int i = 1; i <= d.nx; i++)
            for (int j = 1; j <= d.ny; j++) {
                int id = IDX(d, i, j);
                if (!d.mask[id]) continue;
                u[id] += alpha * p[id];
                r[id] -= alpha * Ap[id];
            }

        double rr = dot_local(d, r, r);
        MPI_Allreduce(MPI_IN_PLACE, &rr, 1, MPI_DOUBLE, MPI_SUM, d.comm);
        if (std::sqrt(rr) < 1e-8) break;

        applyDinv(d, r, z);

        double rz_new = dot_local(d, r, z);
        MPI_Allreduce(MPI_IN_PLACE, &rz_new, 1, MPI_DOUBLE, MPI_SUM, d.comm);

        double beta = rz_new / rz;

#pragma omp parallel for collapse(2)
        for (int i = 1; i <= d.nx; i++)
            for (int j = 1; j <= d.ny; j++) {
                int id = IDX(d, i, j);
                if (!d.mask[id]) continue;
                p[id] = z[id] + beta * p[id];
            }

        rz = rz_new;
    }

    double t1 = MPI_Wtime();

    double rr = dot_local(d, r, r);
    MPI_Allreduce(MPI_IN_PLACE, &rr, 1, MPI_DOUBLE, MPI_SUM, d.comm);

    return { k, std::sqrt(rr), t1 - t0 };
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int M = 40, N = 40;
    if (argc >= 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    }

    Dist d;
    setup_dist(d, M, N, MPI_COMM_WORLD);

    std::vector<double> u((d.nx + 2) * (d.ny + 2), 0.0);
    std::vector<double> f((d.nx + 2) * (d.ny + 2), 0.0);
    d.mask.assign((d.nx + 2) * (d.ny + 2), 0);

    for (int i = 1; i <= d.nx; i++) {
        int I = d.istart + (i - 1);
        double x = d.A1 + I * d.h1;
        for (int j = 1; j <= d.ny; j++) {
            int J = d.jstart + (j - 1);
            double y = d.A2 + J * d.h2;
            int id = IDX(d, i, j);
            if (inDomain(x, y)) {
                d.mask[id] = 1;
                f[id] = 1.0;
            }
        }
    }

    Stats st = solve_pcg(d, u, f);

    if (rank == 0) {
        std::cout << "[summary] iters=" << st.iters
            << ", ||r||=" << std::scientific << st.rnorm
            << ", time=" << std::fixed << st.time << " s\n";
    }

    MPI_Finalize();
    return 0;
}
