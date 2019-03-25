#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <time.h>
#include <math.h>
#include <string.h>

static struct timespec timespec_start, timespec_stop;

struct timespec timespec_diff(struct timespec start, struct timespec stop)
{
	struct timespec temp;
	if ((stop.tv_nsec-start.tv_nsec) < 0) {
		temp.tv_sec = stop.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + stop.tv_nsec - start.tv_nsec;
	} else {
		temp.tv_sec = stop.tv_sec - start.tv_sec;
		temp.tv_nsec = stop.tv_nsec -  start.tv_nsec;
	}
	return temp;
}

void start_timer()
{
    clock_gettime(CLOCK_REALTIME, &timespec_start);
}

double stop_timer()
{
    clock_gettime(CLOCK_REALTIME, &timespec_stop);
    struct timespec timespec_diff_time = timespec_diff(timespec_start, timespec_stop);
    double result = timespec_diff_time.tv_sec + timespec_diff_time.tv_nsec / (double) 1000000000;
    return result;
}

long stop_timer_long()
{
    clock_gettime(CLOCK_REALTIME, &timespec_stop);
    struct timespec timespec_diff_time = timespec_diff(timespec_start, timespec_stop);
    return timespec_diff_time.tv_sec * 1000000000 + timespec_diff_time.tv_nsec;
}


typedef struct matrix {
    double *data;
    int n; //rows
    int m; //columns
} matrix_t;

matrix_t* zeros(int n, int m) 
{
    double *data = calloc(n * m, sizeof(double));
    matrix_t *A = malloc(sizeof(matrix_t));
    A->data = data;
    A->n=n;
    A->m=m;
    return A;
}

gsl_matrix_view create_matrix_view (matrix_t *matrix)
{
    return gsl_matrix_view_array(matrix->data, matrix->n, matrix->m);
}

int get_position (matrix_t *matrix, int i, int j) 
{
    return i*matrix->m + j;
}


int naive_multiplication(matrix_t *A, matrix_t *B, matrix_t *C)
{
    double *Adata = A->data;
    double *Bdata = B->data;
    double *Cdata = C->data;

    int Bm = B->m;
    int An = A->n;
    int Am = A->m;
    int Cm = C->m;

    int i, j, k;
    if (A->n != B->m || A->m != B->n)
        return -1;
    for (j = 0; j < Bm; j++) {
        for (i = 0; i < An; i++) {
            int c_pos = i*Cm + j;
            for (k = 0; k < Am; k++) {
                int a_pos = i*Am + k;
                int b_pos = k*Bm + j;
                Cdata[c_pos] = Cdata[c_pos] + (Adata[a_pos] * Bdata[b_pos]);
            }
        }
    }
    return 0;
}

int better_multiplication(matrix_t *A, matrix_t *B, matrix_t *C)
{
    double *Adata = A->data;
    double *Bdata = B->data;
    double *Cdata = C->data;

    int Bm = B->m;
    int An = A->n;
    int Am = A->m;
    int Cm = C->m;

    int i, j, k;
    if (A->n != B->m || A->m != B->n)
        return -1;
    for (i = 0; i < An; i++) {
        for (k = 0; k < Am; k++) {
            int a_pos = i*Am + k;
            for (j = 0; j < Bm; j++) {
                int c_pos = i*Cm + j;
                int b_pos = k*Bm + j;
                Cdata[c_pos] = Cdata[c_pos] + (Adata[a_pos] * Bdata[b_pos]);
            }
        }
    }
    return 0;
}

void print_matrix(matrix_t *A)
{
    for (int j = 0; j < A->n; j++) {
        printf("[");
        for(int i = 0; i < A->m; i++) {
            int pos = get_position(A, j, i);
            printf(" %6.2f ", A->data[pos]);
        }
        printf("]\n");
    }
}

void fill_matrix(matrix_t *A)
{
    for (int j = 0; j < A->n; j++) {
        for(int i = 0; i < A->m; i++) {
            int pos = get_position(A, j, i);
            A->data[pos] = pos;
        }
    }
}
void fill_matrix_zeros(matrix_t *A)
{
    for (int j = 0; j < A->n; j++) {
        for(int i = 0; i < A->m; i++) {
            int pos = get_position(A, j, i);
            A->data[pos] = 0;
        }
    }
}
void fill_matrix_random(matrix_t *A)
{
    for (int j = 0; j < A->n; j++) {
        for(int i = 0; i < A->m; i++) {
            int pos = get_position(A, i, j);
            A->data[pos] = rand() /(double) RAND_MAX;
        }
    }
}

void free_matrix(matrix_t *A)
{
    free(A->data);
    free(A);
}

void test_all()
{
    FILE* csv = fopen("matrix_multiplication.csv", "w");
    char *buffer = malloc(255 * sizeof(char));
    snprintf(buffer, 255, "size,iteration,naive,better,blas\n");
    fwrite(buffer, 1, strlen(buffer), csv);

    for (int size = 25; size <= 500; size+=25) {
        for (int iteration = 0; iteration < 10; iteration++) {
            matrix_t *A = zeros(size, size);
            fill_matrix_random(A);
            matrix_t *B = zeros(size, size);
            fill_matrix_random(B);
            matrix_t *C = zeros(A->n, B->m);

            fill_matrix_zeros(C);
            start_timer();
            naive_multiplication(A, B, C);
            double naive_result = stop_timer();

            fill_matrix_zeros(C);
            start_timer();
            better_multiplication(A, B, C);
            double better_result = stop_timer();

            fill_matrix_zeros(C);
            gsl_matrix_view v1 = create_matrix_view(A);
            gsl_matrix_view v2 = create_matrix_view(B);
            gsl_matrix_view v3 = create_matrix_view(C);
            start_timer();
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                  1.0, &v1.matrix, &v2.matrix,
                  0.0, &v3.matrix);
            double blas_result = stop_timer();

            free_matrix(A);
            free_matrix(B);
            free_matrix(C);

            snprintf(buffer, 255, "%d,%d,%lf,%lf,%lf\n", size, iteration, naive_result, better_result, blas_result);
            fwrite(buffer, 1, strlen(buffer), csv);
            printf("%s", buffer);
        }
    }

    fclose(csv);
    free(buffer);
}

void test() 
{
    matrix_t *matrix1 = zeros(2, 3);
    fill_matrix(matrix1);
    print_matrix(matrix1);

    printf("\n");

    matrix_t *matrix2 = zeros(3, 2);
    fill_matrix(matrix2);
    print_matrix(matrix2);

    printf("\n");

    matrix_t *matrix3 = zeros(matrix1->n, matrix2->m);
    naive_multiplication(matrix1, matrix2, matrix3);
    print_matrix(matrix3);

    free_matrix(matrix1);
    free_matrix(matrix2);
    free_matrix(matrix3);
}

int main (void)
{
    test_all();      
    return 0;
}
