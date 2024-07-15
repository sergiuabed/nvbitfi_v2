TARGET = matrix_multiplication

# CUDA
NVCC = `which nvcc`
ARCH=61

all: clean $(TARGET) 

$(TARGET): 
#	$(NVCC) -o simple_add -Xptxas -v -arch=sm_$(ARCH) simple_add.cu 
	$(NVCC) -o matrix_multiplication -Xptxas -v -arch=sm_$(ARCH) main.cu kernels.cu utils.cpp

test:
	./matrix_multiplication 

golden:
	./matrix_multiplication >golden_stdout.txt 2>golden_stderr.txt

clean:
	rm -f *.o *~ matrix_multiplication

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
