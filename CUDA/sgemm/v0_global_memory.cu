#include <cstdio>

#define A(i, j) a[i * lda + j]
#define B(i, j) b[i * ldb + j]

void random_matrix(int m, int n, float* a) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
#if 1
            a[i * n + j] = rand() % 100;
#else
            A(i, j) = (j - i) % 3;
#endif
        }
    }
}