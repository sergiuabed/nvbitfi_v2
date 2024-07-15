TARGET = odd_even_sort

# CUDA
NVCC = `which nvcc`
ARCH=61

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o odd_even_sort -arch=sm_$(ARCH) main.cu 

test:
	./odd_even_sort

golden:
	./odd_even_sort >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ odd_even_sort

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
