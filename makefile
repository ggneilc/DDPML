CC = mpicc
FFLAGS = -O3 -Wall 
LFLAGS = -lm -fopenmp

.PHONY: clean

main.exe: matrix.o model.o dataloader.o
	$(CC)  $^ -o $@  $(LFLAGS)


matrix.o: matrix.c
	$(CC) $(FFLAGS) -c $< -fopenmp

model.o: model.c
	$(CC) $(FFLAGS) -c $<

dataloader.o: dataloader.c
	$(CC) $(FFLAGS) -c $<

clean:
	rm -f *.o main.exe
