#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "mvr.h"

const long double PI = 3.141592653589793;
long double OMEGA = 1.9;
long double EPSILON = 1e-7;
int NMAX = 5000000;

// Функция точного решения (вариант 9)
long double u_exact(long double x, long double y) {
    return std::exp(x * x - y * y);
}

// Правая часть f*(x,y) для тестовой задачи (взята так, чтобы u_exact было решением)
long double f_test(long double x, long double y) {
    return -4 * (x * x + y * y) * std::exp(x * x - y * y);
}

// Правая часть f(x,y) для основной задачи (вариант 9)
long double f_base(long double x, long double y) {
    return std::atan(x / y);
}

// Граничные условия

long double mu1(long double y) { return 0.0; } // u(a, y)
long double mu2(long double y) { return 0.0; } // u(b, y)
long double mu3(long double x) { return sin(PI * x) * sin(PI * x); } // u(x, c)
long double mu4(long double x) { return cosh((x - 1) * (x - 2)) - 1.0; } // u(x, d)

using Matrix = std::vector<std::vector<long double>>;

void initialize(Matrix& u, const Grid& grid, bool is_test) {
    int n = grid.n, m = grid.m;
    long double a = grid.a, b = grid.b, c = grid.c, d = grid.d;
    long double h = grid.h, k = grid.k;

    if (is_test) {
        for (int j = 0; j <= m; ++j) {
            long double y = c + j * k;
            u[0][j] = u_exact(a, y);   // левая граница x = a
            u[n][j] = u_exact(b, y);   // правая граница x = b
        }
        for (int i = 0; i <= n; ++i) {
            long double x = a + i * h;
            u[i][0] = u_exact(x, c);   // нижняя граница y = c
            u[i][m] = u_exact(x, d);   // верхняя граница y = d
        }

        // (необязательно, но можно инициализировать внутренность)
        for (int i = 1; i < n; ++i) {
            for (int j = 1; j < m; ++j) {
                long double alpha = i * grid.h;
                u[i][j] = (1 - alpha) * u[0][j] + alpha * u[n][j];
            }
        }
    }
    else {
        // основная задача — с му-функциями
        for (int j = 0; j <= m; ++j) {
            long double y = c + j * k;
            u[0][j] = mu1(y);
            u[n][j] = mu2(y);
        }
        for (int i = 0; i <= n; ++i) {
            long double x = a + i * h;
            u[i][0] = mu3(x);
            u[i][m] = mu4(x);
        }

        for (int i = 1; i < n; ++i)
            for (int j = 1; j < m; ++j)
                u[i][j] = 0.25 * (u[i][0] + u[i][m] + u[0][j] + u[n][j]);
    }
}

// Метод верхней релаксации
int solve(Matrix& u, const Grid& grid, bool is_test_task) {
    int n = grid.n, m = grid.m;
    long double a = grid.a, b = grid.b, c = grid.c, d = grid.d;
    long double h2 = grid.h * grid.h;
    long double k2 = grid.k * grid.k;
    long double denom = 2 * (1 / h2 + 1 / k2);
    int iter = 0;

    while (iter < NMAX) {
        long double max_diff = 0.0;
        for (int i = 1; i < n; ++i) {
            long double x = a + i * grid.h;
            for (int j = 1; j < m; ++j) {
                long double y = c + j * grid.k;
                long double rhs = is_test_task ? f_test(x, y) : f_base(x, y);
                long double u_new = (1 - OMEGA) * u[i][j] + OMEGA * (
                    (u[i + 1][j] + u[i - 1][j]) / h2 +
                    (u[i][j + 1] + u[i][j - 1]) / k2 + rhs) / denom;
                max_diff = std::max(max_diff, std::abs(u_new - u[i][j]));
                u[i][j] = u_new;

            }
        }
        if (iter % 1000 == 0) {
            std::cout << "Итерация " << iter << ", max_diff = " << max_diff << std::endl;
        }
        ++iter;
        if (max_diff < EPSILON) break;
    }
    return iter;
}

// Вычисление погрешности e1 (тестовая задача)
long double compute_error(const Matrix& u_numeric, const Grid& g, long double& max_x, long double& max_y) {
    long double max_diff = 0.0;
    for (int i = 1; i < g.n; ++i) {
        for (int j = 1; j < g.m; ++j) {
            long double x = g.a + i * g.h;
            long double y = g.c + j * g.k;
            long double diff = std::abs(u_exact(x, y) - u_numeric[i][j]);
            if (diff > max_diff) {
                max_diff = diff;
                max_x = x;
                max_y = y;
            }
        }
    }
    return max_diff;
}

// Погрешность между численными решениями (основная задача)
long double compute_difference(const Matrix& u1, const Matrix& u2, const Grid& g1, const Grid& g2, long double& max_x, long double& max_y) {
    long double max_diff = 0.0;
    for (int i = 0; i <= g1.n; ++i) {
        for (int j = 0; j <= g1.m; ++j) {
            long double diff = std::abs(u1[i][j] - u2[2 * i][2 * j]);
            if (diff > max_diff) {
                max_diff = diff;
                max_x = g1.a + i * g1.h;
                max_y = g1.c + j * g1.k;
            }
        }
    }
    return max_diff;
}

// Вычисление нормы невязки (max-норма)
long double compute_residual(const Matrix& u, const Grid& g, bool test) {
    long double max_r = 0.0;
    for (int i = 1; i < g.n; ++i) {
        for (int j = 1; j < g.m; ++j) {
            long double x = g.a + i * g.h;
            long double y = g.c + j * g.k;
            long double f = test ? f_test(x, y) : f_base(x, y);
            long double laplace = (u[i - 1][j] - 2 * u[i][j] + u[i + 1][j]) / (g.h * g.h) +
                (u[i][j - 1] - 2 * u[i][j] + u[i][j + 1]) / (g.k * g.k);
            long double r = std::abs(laplace + f);
            max_r = std::max(max_r, r);
        }
    }
    return max_r;
}

void save_matrix(const Matrix& u, const Grid& grid, const std::string& filename) {
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(8);
    for (int j = grid.m; j >= 0; --j) {
        for (int i = 0; i <= grid.n; ++i) {
            out << u[i][j] << (i < grid.n ? "," : "");
        }
        out << "\n";
    }
    out.close();
}

Grid::Grid(int _n, int _m) {
    a = 0.0;
    b = 1.0;
    c = 0.0;
    d = 1.0;
    n = _n;
    m = _m;
    h = (b - a) / n;
    k = (d - c) / m;
}

void print_table(const Matrix& u, const Grid& grid, const std::string& name) {
    std::cout << "\n" << name << " (" << grid.n + 1 << "x" << grid.m + 1 << "):\n";
    for (int j = grid.m; j >= 0; --j) {
        for (int i = 0; i <= grid.n; ++i) {
            std::cout << std::setw(10) << std::setprecision(6) << u[i][j] << " ";
        }
        std::cout << "\n";
    }
}

// Вывод таблицы точного решения
void print_exact_table(const Grid& g) {
    std::cout << "\nТочное решение u*(x,y):\n";
    for (int j = g.m; j >= 0; --j) {
        for (int i = 0; i <= g.n; ++i) {
            long double x = g.a + i * g.h;
            long double y = g.c + j * g.k;
            std::cout << std::setw(12) << std::setprecision(6) << u_exact(x, y) << " ";
        }
        std::cout << "\n";
    }
}

// Вывод таблицы разности u*(x,y) - v(x,y)
void print_difference_table(const Matrix& u_numeric, const Grid& g) {
    std::cout << "\nРазность u*(x,y) - v(N)(x,y):\n";
    for (int j = g.m; j >= 0; --j) {
        for (int i = 0; i <= g.n; ++i) {
            long double x = g.a + i * g.h;
            long double y = g.c + j * g.k;
            long double diff = u_exact(x, y) - u_numeric[i][j];
            std::cout << std::setw(12) << std::setprecision(6) << diff << " ";
        }
        std::cout << "\n";
    }
}

void print_report(const std::string& task_name, int n, int m, long double omega, long double eps_met, int iter, long double error, long double res, long double max_x, long double max_y, const std::string& type) {
    std::cout << "\n--- Справка: " << task_name << " ---\n";
    std::cout << "Сетка: n = " << n << ", m = " << m << "\n";
    std::cout << "Метод: верхней релаксации\n";
    std::cout << "Параметры: omega = " << omega << ", eps_met = " << eps_met << ", Nmax = " << NMAX << "\n";
    std::cout << "Итераций затрачено: " << iter << "\n";
    std::cout << (type == "e1" ? "Погрешность e1 (тестовая задача): " : "Точность e2 (основная задача): ") << error << "\n";
    std::cout << "Максимум в узле: x = " << max_x << ", y = " << max_y << "\n";
    std::cout << "Норма невязки: ||R(N)|| = " << res << " (max-норма)\n";
    std::cout << "------------------------------\n";
}

int main() {
    setlocale(LC_ALL, "Russian");
    int n, m;
    long double omega, eps_met;

    std::cout << "Введите количество разбиений по x (n): "; std::cin >> n;
    std::cout << "Введите количество разбиений по y (m): "; std::cin >> m;
    std::cout << "Введите параметр метода omega (0 < w < 2): "; std::cin >> omega;
    std::cout << "Введите точность метода eps_met: "; std::cin >> eps_met;

    const_cast<long double&>(OMEGA) = omega;
    const_cast<long double&>(EPSILON) = eps_met;

    // Тестовая задача
    std::vector<int> test_n = { n, 2 * n, 4 * n };
    std::vector<int> test_m = { m, 2 * m, 4 * m };
    std::vector<long double> errors;

    for (size_t idx = 0; idx < test_n.size(); ++idx) {
        Grid grid(test_n[idx], test_m[idx]);
        Matrix u(grid.n + 1, std::vector<long double>(grid.m + 1, 0.0));
        initialize(u, grid, true);
        int iter = solve(u, grid, true);
        long double max_x, max_y;
        long double err = compute_error(u, grid, max_x, max_y);
        errors.push_back(err);
        std::cout << "\n[Проверка сходимости] n = " << test_n[idx] << ", m = " << test_m[idx] << ", e1 = " << err << "\n";
    }

    std::cout << "\nОценка порядка сходимости по e1:\n";
    for (size_t i = 1; i < errors.size(); ++i) {
        long double rate = std::log2(errors[i - 1] / errors[i]);
        std::cout << "Порядок между e1(" << test_n[i - 1] << ") и e1(" << test_n[i] << ") = " << rate << "\n";
    }

    // Тестовая задача для отчёта
    Grid grid1(n, m);
    Matrix u_test(grid1.n + 1, std::vector<long double>(grid1.m + 1, 0.0));
    initialize(u_test, grid1, true);
    int iter_test = solve(u_test, grid1, true);
    long double max_x1, max_y1;
    long double error1 = compute_error(u_test, grid1, max_x1, max_y1);
    long double res1 = compute_residual(u_test, grid1, true);
    print_report("Тестовая задача", n, m, omega, eps_met, iter_test, error1, res1, max_x1, max_y1, "e1");
   // print_exact_table(grid1);
    //print_table(u_test, grid1, "Численное решение v(N)(x,y)");
    //print_difference_table(u_test, grid1);

    // Основная задача
    Grid grid_main(n, m);
    Matrix u_main(grid_main.n + 1, std::vector<long double>(grid_main.m + 1, 0.0));
    initialize(u_main, grid_main, false);
    int iter_main = solve(u_main, grid_main, false);

    Grid grid_main_fine(2 * n, 2 * m);
    Matrix u_main_fine(grid_main_fine.n + 1, std::vector<long double>(grid_main_fine.m + 1, 0.0));
    initialize(u_main_fine, grid_main_fine, false);
    solve(u_main_fine, grid_main_fine, false);

    long double max_x_main, max_y_main;
    long double error_main = compute_difference(u_main, u_main_fine, grid_main, grid_main_fine, max_x_main, max_y_main);
    long double res_main = compute_residual(u_main, grid_main, false);

    print_report("Основная задача", n, m, omega, eps_met, iter_main, error_main, res_main, max_x_main, max_y_main, "e2");
    save_matrix(u_main, grid_main, "main_solution.csv");
    //print_table(u_main, grid_main, "Основная задача");

    return 0;
}