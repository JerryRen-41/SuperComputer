#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#else
static inline double omp_get_wtime() {
    using namespace std::chrono;
    static auto t0 = high_resolution_clock::now();
    auto t = high_resolution_clock::now();
    return duration<double>(t - t0).count();
}
static inline void omp_set_num_threads(int) {}
#endif

using std::vector; using std::string; using std::cout; using std::endl;

struct Problem {
    int M{ 400 }, N{ 600 };                 // 网格分段（结点数=M+1,N+1）
    double A1{ -1.2 }, B1{ 1.2 };         // x ∈ [A1,B1]
    double A2{ -1.2 }, B2{ 1.2 };         // y ∈ [A2,B2]
    double h1{ 0 }, h2{ 0 }, hmax{ 0 }, eps{ 0 };
    vector<vector<double>> w, F, a, b, Ddiag;
};

static inline void init_problem(Problem& P) {
    P.h1 = (P.B1 - P.A1) / P.M;
    P.h2 = (P.B2 - P.A2) / P.N;
    P.hmax = std::max(P.h1, P.h2);
    P.eps = P.hmax * P.hmax;                  // ε = h^2
    P.w.assign(P.M + 1, vector<double>(P.N + 1, 0.0));
    P.F.assign(P.M + 1, vector<double>(P.N + 1, 0.0));
    P.a.assign(P.M + 1, vector<double>(P.N + 1, 0.0));
    P.b.assign(P.M + 1, vector<double>(P.N + 1, 0.0));
    P.Ddiag.assign(P.M + 1, vector<double>(P.N + 1, 0.0));
}

// 区间交长度
static inline double overlap_len(double L0, double L1, double R0, double R1) {
    double lo = std::max(L0, R0), hi = std::min(L1, R1);
    return (hi > lo) ? (hi - lo) : 0.0;
}

// 几何落盘：F、a、b（方案 B）
static void build_coeff_rhs(Problem& P) {
    const double h1 = P.h1, h2 = P.h2, eps = P.eps;

    // F_ij：cell 在 D = (-1,1)^2 \ (0,1)^2 中的面积比例
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < P.M; ++i) {
        for (int j = 1; j < P.N; ++j) {
            double xL = P.A1 + (i - 0.5) * h1, xR = P.A1 + (i + 0.5) * h1;
            double yB = P.A2 + (j - 0.5) * h2, yT = P.A2 + (j + 0.5) * h2;

            double wx_sq = overlap_len(xL, xR, -1.0, 1.0);
            double wy_sq = overlap_len(yB, yT, -1.0, 1.0);
            double S_in_square = wx_sq * wy_sq;

            double wx_q1 = overlap_len(xL, xR, 0.0, 1.0);
            double wy_q1 = overlap_len(yB, yT, 0.0, 1.0);
            double S_in_q1 = wx_q1 * wy_q1;

            double S_D = std::max(0.0, S_in_square - S_in_q1);
            P.F[i][j] = S_D / (h1 * h2); // f=1(in D), 0(in D^)
        }
    }

    // a_ij, b_ij：沿半整点边对 k 积分/边长；k=1(in D), 1/eps(in D^)
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < P.M; ++i) {
        for (int j = 1; j < P.N; ++j) {
            // 垂直边 x = x_{i-1/2}
            double x_v = P.A1 + (i - 0.5) * h1;
            double yB = P.A2 + (j - 0.5) * h2, yT = P.A2 + (j + 0.5) * h2;
            double ly_sq = overlap_len(yB, yT, -1.0, 1.0);
            double ly_q1 = (x_v > 0.0 && x_v < 1.0) ? overlap_len(yB, yT, 0.0, 1.0) : 0.0;
            double lD_v = std::max(0.0, ly_sq - ly_q1);
            double integ_a = lD_v * 1.0 + (h2 - lD_v) * (1.0 / eps);
            P.a[i][j] = integ_a / h2;

            // 水平边 y = y_{j-1/2}
            double y_h = P.A2 + (j - 0.5) * h2;
            double xL = P.A1 + (i - 0.5) * h1, xR = P.A1 + (i + 0.5) * h1;
            double lx_sq = overlap_len(xL, xR, -1.0, 1.0);
            double lx_q1 = (y_h > 0.0 && y_h < 1.0) ? overlap_len(xL, xR, 0.0, 1.0) : 0.0;
            double lD_h = std::max(0.0, lx_sq - lx_q1);
            double integ_b = lD_h * 1.0 + (h1 - lD_h) * (1.0 / eps);
            P.b[i][j] = integ_b / h1;
        }
    }

    // 预条件子 D 的对角元
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < P.M; ++i) {
        for (int j = 1; j < P.N; ++j) {
            P.Ddiag[i][j] = ((P.a[i + 1][j] + P.a[i][j]) / (h1 * h1))
                + ((P.b[i][j + 1] + P.b[i][j]) / (h2 * h2));
            if (P.Ddiag[i][j] <= 0) P.Ddiag[i][j] = 1.0;
        }
    }
}

// 离散算子 A
static void applyA(const Problem& P, const vector<vector<double>>& w, vector<vector<double>>& Aw) {
    const double h1 = P.h1, h2 = P.h2;
    Aw.assign(P.M + 1, vector<double>(P.N + 1, 0.0));
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < P.M; ++i) {
        for (int j = 1; j < P.N; ++j) {
            double fxp = P.a[i + 1][j] * (w[i + 1][j] - w[i][j]);
            double fxm = P.a[i][j] * (w[i][j] - w[i - 1][j]);
            double fyp = P.b[i][j + 1] * (w[i][j + 1] - w[i][j]);
            double fym = P.b[i][j] * (w[i][j] - w[i][j - 1]);
            Aw[i][j] = -((fxp - fxm) / (h1 * h1) + (fyp - fym) / (h2 * h2));
        }
    }
}

static double dotE(const Problem& P, const vector<vector<double>>& a, const vector<vector<double>>& b) {
    double s = 0.0;
#pragma omp parallel for collapse(2) reduction(+:s) schedule(static)
    for (int i = 1; i < P.M; ++i)
        for (int j = 1; j < P.N; ++j)
            s += a[i][j] * b[i][j];
    return s * P.h1 * P.h2;
}

static void applyDinv(const Problem& P, const vector<vector<double>>& r, vector<vector<double>>& z) {
    z.assign(P.M + 1, vector<double>(P.N + 1, 0.0));
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < P.M; ++i)
        for (int j = 1; j < P.N; ++j)
            z[i][j] = r[i][j] / P.Ddiag[i][j];
}

struct Stats { int iters{ 0 }; double resE{ 0 }, dxE{ 0 }, time{ 0 }; };

static Stats solve_pcg(Problem& P, int maxIter, double eps_rel, double delta, int report_every) {
    Stats st{};
    vector<vector<double>> r(P.M + 1, vector<double>(P.N + 1, 0.0)), z, p, Ap, w_prev;

    r = P.F;                               
    double t0 = omp_get_wtime();
    double rr0 = dotE(P, r, r), norm0 = std::sqrt(rr0);
    if (norm0 == 0.0) { st.iters = 0; st.resE = 0; st.time = 0; st.dxE = 0; return st; }

    applyDinv(P, r, z); p = z; double rz_old = dotE(P, r, z);

    for (int k = 0; k < maxIter; ++k) {
        w_prev = P.w;
        applyA(P, p, Ap);
        double pAp = dotE(P, p, Ap);
        if (std::fabs(pAp) < 1e-30) break;
        double alpha = rz_old / pAp;

#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < P.M; ++i)
            for (int j = 1; j < P.N; ++j) {
                P.w[i][j] += alpha * p[i][j];
                r[i][j] -= alpha * Ap[i][j];
            }

        double rr = dotE(P, r, r), norm = std::sqrt(rr);
        st.iters = k + 1;

        vector<vector<double>> dx(P.M + 1, vector<double>(P.N + 1, 0.0));
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < P.M; ++i)
            for (int j = 1; j < P.N; ++j)
                dx[i][j] = P.w[i][j] - w_prev[i][j];
        st.dxE = std::sqrt(dotE(P, dx, dx));

        if (report_every > 0 && (k % report_every == 0)) {
            cout << "iter " << std::setw(4) << k
                << "  ||r||_E=" << std::scientific << std::setprecision(6) << norm
                << " (rel=" << (norm / norm0) << ")"
                << "  ||dw||_E=" << std::fixed << std::setprecision(6) << st.dxE << "\n";
        }

        if ((norm <= eps_rel * norm0) || (st.dxE < delta)) { st.resE = norm; break; }

        applyDinv(P, r, z);
        double rz = dotE(P, r, z); double beta = rz / rz_old; rz_old = rz;
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i < P.M; ++i)
            for (int j = 1; j < P.N; ++j)
                p[i][j] = z[i][j] + beta * p[i][j];
    }
    st.time = omp_get_wtime() - t0;
    if (st.resE == 0.0) st.resE = std::sqrt(dotE(P, r, r));
    return st;
}

// 保存到当前目录
static void save_like_sample(const Problem& P, const std::string& fname) {
    std::ofstream ofs(fname);
    if (!ofs) { std::cerr << "Cannot open " << fname << " for write\n"; return; }
    ofs.setf(std::ios::fixed); ofs << std::setprecision(8);
    // 行优先：外层 j(0..N)，行内 i=0..M
    for (int j = 0; j <= P.N; ++j) {
        for (int i = 0; i <= P.M; ++i) {
            ofs << P.w[i][j];
            if (i < P.M) ofs << "    ";
        }
        ofs << '\n';
    }
}

static Stats run_case(int M, int N, int threads, int report_every,
    bool save_solution = true) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    Problem P; P.M = M; P.N = N;
    init_problem(P); build_coeff_rhs(P);
    cout << "=== Grid " << M << "x" << N << "  Threads=" << threads
        << "  (eps=" << P.eps << ") ===\n";

    Stats st = solve_pcg(P, /*maxIter*/8000, /*eps_rel*/1e-6, /*delta*/1e-8, report_every);

    cout << "[summary] iters=" << st.iters
        << ", ||r||_E=" << std::scientific << std::setprecision(6) << st.resE
        << ", ||dw||_E=" << std::fixed << std::setprecision(6) << st.dxE
        << ", time=" << std::fixed << std::setprecision(6) << st.time << " s\n";

    if (save_solution) {
        char name[128];
        std::snprintf(name, sizeof(name), "result_%dx%d_t%d.txt", M, N, threads);
        save_like_sample(P, name);
        cout << "Saved solution to: " << name << "\n";
    }
    return st;
}

int main(int argc, char** argv) {
    // 单次模式：./app M N threads [report_every]
    if (argc >= 4) {
        int M = std::atoi(argv[1]), N = std::atoi(argv[2]), threads = std::atoi(argv[3]);
        int report_every = (argc >= 5) ? std::atoi(argv[4]) : 200;
        run_case(M, N, threads, report_every, /*save_solution=*/true);
        return 0;
    }

    // 批量模式：两组尺寸
    // 400x600 -> 2/4/8/16
    // 800x1200 -> 4/8/16/32
    std::vector<std::pair<int, int>> sizes = { {400,600}, {800,1200} };
    std::map<std::pair<int, int>, std::vector<int>> threads_map = {
        {{400,600},  {2,4,8,16}},
        {{800,1200}, {4,8,16,32}}
    };

    cout << "\n>>> Batch run (multiple grids):\n";
    cout << std::left << std::setw(12) << "Grid"
        << std::setw(10) << "Threads"
        << std::setw(10) << "Iters"
        << std::setw(14) << "Time(s)"
        << std::setw(14) << "||r||_E" << '\n';
    cout << std::string(60, '-') << '\n';

    for (auto sz : sizes) {
        int M = sz.first, N = sz.second;
        for (int t : threads_map[sz]) {
            auto st = run_case(M, N, t, /*report_every=*/0, /*save_solution=*/true);
            cout << std::left
                << std::setw(12) << (std::to_string(M) + "x" + std::to_string(N))
                << std::setw(10) << t
                << std::setw(10) << st.iters
                << std::setw(14) << std::fixed << std::setprecision(6) << st.time
                << std::setw(14) << std::scientific << std::setprecision(6) << st.resE
                << '\n';
        }
    }
    return 0;

}
