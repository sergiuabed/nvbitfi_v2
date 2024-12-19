TARGET = PCAHyperspectralClassifier

# CUDA
NVCC = `which nvcc`
ARCH=61

all: clean $(TARGET) 

$(TARGET): 
#	$(NVCC) -o simple_add -Xptxas -v -arch=sm_$(ARCH) simple_add.cu 
	$(NVCC) -Xcompiler -fPIC -shared -o pca.so main.cpp kernel_pca.cu -lcublas -lm -lgsl -lgslcblas

test:
#	./simple_add 
#	python3 inference.py --cuda 0 --image IndianPines --checkpoint models/ip/10_IP.pth --model li --pca 10
#	python3 inference.py --cuda 0 --image IndianPines --checkpoint $(LI_MODEL_CHECKPOINTS_PATH)/ip/10_IP.pth --model li --pca 10
	python3 inference.py --cuda 0 --image PaviaU --checkpoint $(LI_MODEL_CHECKPOINTS_PATH)/pu/10_PU.pth --model li --pca 10
#	python3 inference.py --cuda 0 --image PaviaU --checkpoint models/pu/10_PU.pth --model li --pca 10
#	python3 inference.py --cuda 0 --image Salinas --checkpoint $(LI_MODEL_CHECKPOINTS_PATH)/sa/li_et_al_Salinas_K=10.pth --model li --pca 10
#	python3 inference.py --cuda 0 --image IndianPines --checkpoint $(LI_MODEL_CHECKPOINTS_PATH)/ip/7_IP.pth --model li --pca 7
#	python3 inference.py --cuda 0 --image PaviaU --checkpoint $(LI_MODEL_CHECKPOINTS_PATH)/pu/7_PU.pth --model li --pca 7
#	python3 inference.py --cuda 0 --image Salinas --checkpoint $(LI_MODEL_CHECKPOINTS_PATH)/sa/li_et_al_Salinas_K=7.pth --model li --pca 7

golden:
#	./simple_add >golden_stdout.txt 2>golden_stderr.txt
#	python3 inference.py --cuda 0 --image IndianPines --checkpoint models/ip/10_IP.pth --model li --pca 10 >golden_stdout.txt 2>golden_stderr.txt
	python3 inference.py --cuda 0 --image PaviaU --checkpoint models/pu/10_PU.pth --model li --pca 10 >golden_stdout.txt 2>golden_stderr.txt
#	python3 inference.py --cuda 0 --image IndianPines --checkpoint $(LI_MODEL_CHECKPOINTS_PATH)/ip/10_IP.pth --model li --pca 10 >golden_stdout.txt 2>golden_stderr.txt
#	python3 inference.py --cuda 0 --image Salinas --checkpoint models/sa/li_et_al_Salinas_K=10.pth --model li --pca 10 >golden_stdout.txt 2>golden_stderr.txt
#	python3 inference.py --cuda 0 --image IndianPines --checkpoint $(LI_MODEL_CHECKPOINTS_PATH)/ip/7_IP.pth --model li --pca 7 >golden_stdout.txt 2>golden_stderr.txt
#	python3 inference.py --cuda 0 --image PaviaU --checkpoint $(LI_MODEL_CHECKPOINTS_PATH)/pu/7_PU.pth --model li --pca 7 >golden_stdout.txt 2>golden_stderr.txt
#	python3 inference.py --cuda 0 --image Salinas --checkpoint $(LI_MODEL_CHECKPOINTS_PATH)/sa/li_et_al_Salinas_K=7.pth --model li --pca 7 >golden_stdout.txt 2>golden_stderr.txt

#	remove non-deterministic lines from 'golden_stderr.txt' (due to tqdm printing on stderr)
#	IMPORTANT: 'tr' command replaces carriage returns with new lines (due to tqdm separating lines with carriage returns, while grep needs new lines)
	tr '\r' '\n' <golden_stderr.txt | grep -v Inference >grepout
	mv prediction_inference.tif golden_prediction_inference.tif

clean:
	rm -f *.so *~ pca
	rm prediction_inference.tif

clobber: clean
	rm -f golden* *_regcount.p stdout* stderr*
