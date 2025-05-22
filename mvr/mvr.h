#pragma once
#include <vector>

// Глобальные параметры (теперь переменные, а не const)
extern long double OMEGA;
extern long double EPSILON;
extern int NMAX;

// Структура сетки
struct Grid {
    long double a, b, c, d;
    int n, m;
    long double h, k;
    Grid(int _n, int _m);
};

using Matrix = std::vector<std::vector<long double>>;

void initialize(Matrix& u, const Grid& grid);
int solve(Matrix& u, const Grid& grid, bool is_test_task = true);
long double compute_error(const Matrix& u, const Grid& grid);
long double compute_difference(const Matrix& u1, const Matrix& u2, const Grid& g1, const Grid& g2);

