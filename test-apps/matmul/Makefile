TARGET = matmul

# CUDA
NVCC = `which nvcc`
ARCH=61

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o matmul -Xptxas -v -arch=sm_$(ARCH) main.cu 

test:
	./matmul 

golden:
	./matmul >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ matmul

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
