#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <chrono>

using namespace std;

struct Grid {
    int M, N;
    double A1, B1, A2, B2;
    double h1, h2;
    vector<vector<double>> w, f;
    vector<vector<int>> mask;
};

static bool inDomain(double x, double y) {
    if (x < -1.0 || x > 1.0 || y < -1.0 || y > 1.0) return false;
    if (x > 0.0 && y > 0.0) return false;
    return true;
}

static void initialize(Grid& g) {
    g.h1 = (g.B1 - g.A1) / g.M;
    g.h2 = (g.B2 - g.A2) / g.N;

    g.w.assign(g.M + 1, vector<double>(g.N + 1, 0.0));
    g.f.assign(g.M + 1, vector<double>(g.N + 1, 0.0));
    g.mask.assign(g.M + 1, vector<int>(g.N + 1, 0));

    for (int i = 0; i <= g.M; i++) {
        double x = g.A1 + i * g.h1;
        for (int j = 0; j <= g.N; j++) {
            double y = g.A2 + j * g.h2;

            if (i > 0 && i < g.M && j > 0 && j < g.N && inDomain(x, y)) {
                g.mask[i][j] = 1;
                g.f[i][j] = 1.0;
            }
            else {
                g.mask[i][j] = 0;
                g.f[i][j] = 0.0;
                g.w[i][j] = 0.0;
            }
        }
    }
}

static void applyA_to(const Grid& g, const vector<vector<double>>& in,
    vector<vector<double>>& out) {
    if (out.size() != (size_t)(g.M + 1) ||
        (out.size() > 0 && out[0].size() != (size_t)(g.N + 1)))
    {
        out.assign(g.M + 1, vector<double>(g.N + 1, 0.0));
    }
    else {
        for (int i = 0; i <= g.M; i++)
            for (int j = 0; j <= g.N; j++)
                out[i][j] = 0.0;
    }

    double hx2 = g.h1 * g.h1;
    double hy2 = g.h2 * g.h2;
    double diag = 2.0 / hx2 + 2.0 / hy2;

    for (int i = 1; i < g.M; i++) {
        for (int j = 1; j < g.N; j++) {
            if (!g.mask[i][j]) continue;

            double uij = in[i][j];
            double v = diag * uij;

            if (g.mask[i - 1][j]) v -= in[i - 1][j] / hx2;
            if (g.mask[i + 1][j]) v -= in[i + 1][j] / hx2;
            if (g.mask[i][j - 1]) v -= in[i][j - 1] / hy2;
            if (g.mask[i][j + 1]) v -= in[i][j + 1] / hy2;

            out[i][j] = v;
        }
    }
}

static double dotE(const Grid& g,
    const vector<vector<double>>& a,
    const vector<vector<double>>& b) {
    double w = g.h1 * g.h2;
    double s = 0.0;
    for (int i = 0; i <= g.M; i++) {
        for (int j = 0; j <= g.N; j++) {
            if (!g.mask[i][j]) continue;
            s += a[i][j] * b[i][j] * w;
        }
    }
    return s;
}

static void saveResult(const Grid& g, const string& fn) {
    ofstream out(fn);
    out << setprecision(10);
    for (int i = 0; i <= g.M; i++) {
        double x = g.A1 + i * g.h1;
        for (int j = 0; j <= g.N; j++) {
            double y = g.A2 + j * g.h2;
            out << x << " " << y << " " << g.w[i][j] << "\n";
        }
        out << "\n";
    }
    cout << "Saved: " << fn << "\n";
}

struct SolveStats {
    int iters;
    double seconds;
    double final_resE;
};

static SolveStats solve_cg_seq(Grid& g,
    int maxIter = 20000,
    double eps = 1e-6,
    int report_every = 200) {
    vector<vector<double>> r = g.f;
    vector<vector<double>> p = r;
    vector<vector<double>> Ap;

    auto t0 = chrono::high_resolution_clock::now();

    double rr = dotE(g, r, r);
    double norm = sqrt(rr);
    SolveStats st{ 0, 0.0, norm };

    if (norm < eps) {
        st.seconds = chrono::duration<double>(
            chrono::high_resolution_clock::now() - t0).count();
        return st;
    }

    for (int k = 0; k < maxIter; k++) {
        applyA_to(g, p, Ap);
        double pAp = dotE(g, p, Ap);

        if (!std::isfinite(pAp) || fabs(pAp) < 1e-30) {
            cerr << "CG breakdown: pAp = " << pAp
                << " at iter " << k << endl;
            break;
        }

        double alpha = rr / pAp;

        vector<vector<double>> r_new = r;

        for (int i = 1; i < g.M; i++) {
            for (int j = 1; j < g.N; j++) {
                if (!g.mask[i][j]) continue;
                g.w[i][j] += alpha * p[i][j];
                r_new[i][j] -= alpha * Ap[i][j];
            }
        }

        double rr_new = dotE(g, r_new, r_new);

        if (!std::isfinite(rr_new)) {
            cerr << "CG breakdown: rr_new = " << rr_new
                << " at iter " << k << endl;
            break;
        }

        norm = sqrt(rr_new);
        st.iters = k + 1;
        st.final_resE = norm;

        if (norm < eps) break;

        double beta = rr_new / rr;

        for (int i = 1; i < g.M; i++) {
            for (int j = 1; j < g.N; j++) {
                if (!g.mask[i][j]) continue;
                p[i][j] = r_new[i][j] + beta * p[i][j];
            }
        }

        r.swap(r_new);
        rr = rr_new;

        if (report_every > 0 && (k + 1) % report_every == 0) {
            cout << "  iter " << (k + 1)
                << ", ||r||_E = " << norm << "\n";
        }
    }

    st.seconds = chrono::duration<double>(
        chrono::high_resolution_clock::now() - t0).count();

    return st;
}

int main() {
    cout << "Start sequential PCG ..." << endl;

    vector<pair<int, int>> grids = {
    {400, 600},
    {800, 1200}
};

    for (auto& pr : grids) {
        int M = pr.first, N = pr.second;
        Grid g;
        g.M = M; g.N = N;
        g.A1 = -1.0; g.B1 = 1.0;
        g.A2 = -1.0; g.B2 = 1.0;

        initialize(g);
        cout << "Grid " << M << "x" << N << "\n";

        SolveStats st = solve_cg_seq(g, 20000, 1e-6, 200);
        cout << "iters=" << st.iters
            << "  res=" << st.final_resE
            << "  time=" << st.seconds << " s\n";

        ostringstream oss;
        oss << "result_" << M << "x" << N << ".txt";
        saveResult(g, oss.str());
    }

    cout << "Done." << endl;
    return 0;
}
