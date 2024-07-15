TARGET = matmul_float

# CUDA
NVCC = `which nvcc`
ARCH=61

all: clean $(TARGET) 

$(TARGET): 
	$(NVCC) -o matmul_float -Xptxas -v -arch=sm_$(ARCH) main.cu 

test:
	./matmul_float 

golden:
	./matmul_float >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ matmul_float

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
