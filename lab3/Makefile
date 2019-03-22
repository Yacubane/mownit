main : main.c
	gcc main.c -o main -lgsl -lgslcblas -lm -lrt -O0

test : main
	./main

clean :
	rm -rf main