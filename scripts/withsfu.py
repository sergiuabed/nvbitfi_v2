apps = {
	#'simple_add': [
	#		NVBITFI_HOME + '/test-apps/simple_add', # workload directory
	#		'simple_add', # binary name
	#		NVBITFI_HOME + '/test-apps/simple_add/', # path to the binary file
	#		1, # expected runtime
	#		"" # additional parameters to the run.sh
	#	],
	'WS_backprop': [
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/backprop', # workload directory
			'backprop_cuda', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/backprop', # path to the binary file
			2.5, # expected runtime
			"65536" # additional parameters to the run.sh
		],
	'WS_NN': [
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/nn', # workload directory
			'nn', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/nn', # path to the binary file
			1.5, # expected runtime
			NVBITFI_HOME +"/test-apps/rodinia_SFU/data/nn/list640k_64.txt -r 100 -lat 30.0 -lng 90.0" # additional parameters to the run.sh
		],
	'WS_gaussian': [
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/gaussian', # workload directory
			'gaussian', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/gaussian', # path to the binary file
			2.5 ,# expected runtime
			"-f "+NVBITFI_HOME +"/test-apps/rodinia_SFU/data/gaussian/matrix1024.txt" # additional parameters to the run.sh
		],
	'WS_cfd': [
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/cfd', # workload directory
			'euler3d', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/cfd', # path to the binary file
			3.5, # expected runtime
			NVBITFI_HOME +"/test-apps/rodinia_SFU/data/cfd/fvcorr.domn.097K" # additional parameters to the run.sh
		],
	'WS_heartwall': [
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/heartwall', # workload directory
			'heartwall', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/heartwall', # path to the binary file
			1, # expected runtime
			NVBITFI_HOME +"/test-apps/rodinia_SFU/data/heartwall/test.avi 1" # additional parameters to the run.sh
		],
	'WS_lavaMD': [
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/lavaMD', # workload directory
			'lavaMD', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/lavaMD', # path to the binary file
			4, # expected runtime
			"-boxes1d 4" # additional parameters to the run.sh
		],
	'WS_myocyte': [
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/myocyte', # workload directory
			'myocyte', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/myocyte', # path to the binary file
			2, # expected runtime
			"100 1 0" # additional parameters to the run.sh
		],
	'WS_lud': [
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/lud', # workload directory
			'lud', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/lud', # path to the binary file
			2, # expected runtime
			"-s 256 -v" # additional parameters to the run.sh
		],
	'WS_srad_v1': [
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/srad_v1', # workload directory
			'srad', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_SFU/cuda/srad_v1', # path to the binary file
			2, # expected runtime
			"100 0.5 502 458" # additional parameters to the run.sh
		],
}