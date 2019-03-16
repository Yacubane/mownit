#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <time.h>
#include <math.h>
#include <string.h>

const int MAX_MATRIX_SIZE = 10000;
const int MAX_VECTOR_SIZE = 1000000000;

static struct timespec timespec_start, timespec_stop;

struct timespec timespec_diff(struct timespec start, struct timespec stop)
{
	struct timespec temp;
	if ((stop.tv_nsec-start.tv_nsec) < 0) 
    {
		temp.tv_sec = stop.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + stop.tv_nsec - start.tv_nsec;
	} else 
    {
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

double* generate_double_matrix(int size) {
    double* matrix = malloc(size * size * sizeof(double));
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
             matrix[i * size + j] = i * size + j;
        }
    }
    return matrix;
}

double* generate_double_vector(int size) {
    double* vector = malloc(size * sizeof(double));
    for(int i = 0; i < size; i++) {
        vector[i] = i;
    }
    return vector;
}


double* generate_empty_double_vector(int size) {
    double* vector = malloc(size * sizeof(double));
    for(int i = 0; i < size; i++) {
        vector[i] = 0;
    }
    return vector;
}

double vector_ddot(gsl_vector_view v1, gsl_vector_view v2, double* result) {
    start_timer();
    gsl_blas_ddot(&v1.vector, &v2.vector, result);
    return stop_timer();
}

double matrix_dgemv(gsl_matrix_view m1, gsl_vector_view v1, gsl_vector_view v_result) {
    start_timer();
    //y := alpha*A*x + beta*y,  
    gsl_blas_dgemv(CblasNoTrans, 1.0, &m1.matrix, &v1.vector, 1.0, &v_result.vector);
    return stop_timer();
}

double ddot_test(double *double_vector1, double *double_vector2, int size, double *result) {
    gsl_vector_view vector1 = gsl_vector_view_array(double_vector1, size);
    gsl_vector_view vector2 = gsl_vector_view_array(double_vector2, size);

    return vector_ddot(vector1, vector2, result);
}
double dgemv_test(double *double_matrix, double *double_vector, int size, double *double_vector_result) {
    gsl_matrix_view matrix = gsl_matrix_view_array(double_matrix, size, size);
    gsl_vector_view vector = gsl_vector_view_array(double_vector, size);
    gsl_vector_view vector_result = gsl_vector_view_array(double_vector_result, size);

    return matrix_dgemv(matrix, vector, vector_result);
}

void test_vector() {
    double* double_vector = generate_double_vector(MAX_VECTOR_SIZE);
    gsl_vector_view vector = gsl_vector_view_array(double_vector, 2);

    double scalar_result = 0;

    FILE *csv = fopen("vector_text.csv", "w");
    char *buffer = malloc(255 * sizeof(char));
    snprintf(buffer, 255, "size,iteration,result\n");
    fwrite(buffer, 1, strlen(buffer), csv);

    int measurements = 50;
    int block_size = MAX_VECTOR_SIZE / measurements;
    int size = block_size;
    while(size <= MAX_VECTOR_SIZE) {
        for(int iteration = 0; iteration < 10; iteration++) {
            double result = ddot_test(double_vector, double_vector, size, &scalar_result);
            snprintf(buffer, 255, "%d,%d,%lf\n", size, iteration, result);
            printf("%s\n", buffer);
            int res = fwrite(buffer, 1, strlen(buffer), csv);
        }    
        size+=block_size;  
    }

    fclose(csv);
    free(buffer);
    free(double_vector);
}

void test_matrix() {
    double *double_matrix = generate_double_matrix(MAX_MATRIX_SIZE);
    double *double_vector = generate_double_vector(MAX_MATRIX_SIZE);
    double *double_result_vector = generate_empty_double_vector(MAX_MATRIX_SIZE);

    FILE* csv = fopen("matrix_test.csv", "w");
    char *buffer = malloc(255 * sizeof(char));
    snprintf(buffer, 255, "size,iteration,result\n");
    fwrite(buffer, 1, strlen(buffer), csv);
    int size = 100;
    while(size <= MAX_MATRIX_SIZE) {
        for(int iteration = 0; iteration < 10; iteration++) {
            double result = dgemv_test(double_matrix, double_vector, size, double_result_vector);
            snprintf(buffer, 255, "%d,%d,%lf\n", size, iteration, result);
            int res = fwrite(buffer, 1, strlen(buffer), csv);
            printf("%s\n", buffer);
        }
        size += 100;
    }
    fclose(csv);
    free(buffer);
    free(double_matrix);
    free(double_vector);
    free(double_result_vector);
}

void test_all() {
    double *double_matrix = generate_double_matrix(MAX_MATRIX_SIZE);
    double *double_vector = generate_double_vector(MAX_VECTOR_SIZE);

    double *double_result_vector = generate_empty_double_vector(MAX_MATRIX_SIZE);
    double scalar_result = 0;

    FILE* csv = fopen("all_test.csv", "w");
    char *buffer = malloc(255 * sizeof(char));
    snprintf(buffer, 255, "size,iteration,vector_time,matrix_time\n");
    fwrite(buffer, 1, strlen(buffer), csv);
    int size = 100;
    while(size <= MAX_MATRIX_SIZE) {
        for(int iteration = 0; iteration < 10; iteration++) {
            double result1 = ddot_test(double_vector, double_vector, size*size, &scalar_result);
            double result2 = dgemv_test(double_matrix, double_vector, size, double_result_vector);
            snprintf(buffer, 255, "%d,%d,%lf,%lf\n", size*size, iteration, result1, result2);
            int res = fwrite(buffer, 1, strlen(buffer), csv);
            printf("%s\n", buffer);
        }
        size += 100;
    }
    fclose(csv);
    free(buffer);
    free(double_matrix);
    free(double_vector);
    free(double_result_vector);
}

int main (void)
{
    test_all();      
    return 0;
}
