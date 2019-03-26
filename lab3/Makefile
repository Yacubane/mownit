main : main.c
	gcc main.c -o main -lgsl -lgslcblas -lm -lrt -O0

test : main
	gcc main.c -o main -lgsl -lgslcblas -lm -lrt -O0
	./main matrix_multiplication_O0.csv
	gcc main.c -o main -lgsl -lgslcblas -lm -lrt -O3
	./main matrix_multiplication_O3.csv


clean :
	rm -rf main