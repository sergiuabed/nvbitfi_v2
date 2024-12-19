# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os, sys

PYTHON_P = "python"

TIMEOUT_THRESHOLD = 10 # 10X usual runtime 

if 'NVBITFI_HOME' not in os.environ:
	print ("Error: Please set NVBITFI_HOME environment variable")
	sys.exit(-1)
NVBITFI_HOME = os.environ['NVBITFI_HOME']

# verbose = True
verbose = False

detectors = True

# Keep per-app injection logs: This can be helpful for debugging. If this flag
# is set to false, per-injection logs will be deleted. A detailed summary will
# be captured in the results file. 
keep_logs = True

#########################################################################
# Number of injections per app, per instruction group (IGID), per bit-flip
# model (BFM)
# 
# According to http://www.surveysystem.com/sscalc.htm:
# - Confdience interval at 99% condence level is at most 5% with 644 injections
# - Confdience interval at 95% condence level is at most 5% with 384 injections
# - Confdience interval at 95% condence level is at most 3.1% with 1000 injections
# - Confdience interval at 95% condence level is at most 2.19% with 2000 injections
# - Confdience interval at 99% condence level is at most 2.88% with 2000 injections
#########################################################################

# Specify the number of injection sites to create before starting the injection
# campaign. This is essentially the maximum number of injections one can run
# per instruction group (IGID) and bit-flip model (BFM).
# 
# NUM_INJECTIONS = 644
NUM_INJECTIONS = 1000

# Specify how many injections you want to perform per IGID and BFM combination. 
# Only the first THRESHOLD_JOBS will be selected from the generated NUM_INJECTIONS.
#
# THRESHOLD_JOBS = 384
#THRESHOLD_JOBS = 1000
# THRESHOLD_JOBS = 1
# THRESHOLD_JOBS = 25
THRESHOLD_JOBS = 200#1000

# THRESHOLD_JOBS sould be <= NUM_INJECTIONS
assert THRESHOLD_JOBS <= NUM_INJECTIONS


#######################################################################
# Specify library paths
#######################################################################
INJECTOR_LIB = os.environ['NVBITFI_HOME'] + "/injector/injector.so"
PROFILER_LIB = os.environ['NVBITFI_HOME'] + "/profiler/profiler.so"


#######################################################################
# Three injection modes
#######################################################################
RF_MODE = "rf"
INST_VALUE_MODE = "inst_value"
INST_ADDRESS_MODE = "inst_address"

#######################################################################
# Categories of instruction types (IGIDs): This should match the values set in
# arch.h in the nvbitfi/common/
#######################################################################
G_FP64 = 0
G_FP32 = 1
G_LD = 2
G_PR = 3
G_NODEST = 4 # not really an igid
G_OTHERS = 5 
G_GPPR = 6 # instructions that write to either a GPR or a PR register
G_GP = 7 # instructions that write to a GPR register
NUM_INST_GROUPS = 8

IGID_STR = [ "fp64", "fp32", "ld", "pr", "nodest", "others", "gppr", "gp" ]


#######################################################################
# Types of avaialble error models (bit-flip model, BFM): This should match the
# values set in err_injector/error_injector.h. 
#######################################################################
FLIP_SINGLE_BIT = 0
FLIP_TWO_BITS = 1
RANDOM_VALUE = 2
ZERO_VALUE = 3

EM_STR = [ "FLIP_SINGLE_BIT", "FLIP_TWO_BITS", "RANDOM_VALUE", "ZERO_VALUE"]

rf_inst = ""

#######################################################################
# Categories of error injection outcomes
#######################################################################
# Masked
MASKED_NOT_READ = 1
MASKED_WRITTEN = 2
MASKED_OTHER = 3

# DUEs
TIMEOUT = 4
NON_ZERO_EC = 5 # non zero exit code

# Potential DUEs with appropriate detectors in place
MASKED_KERNEL_ERROR = 6
SDC_KERNEL_ERROR = 7
NON_ZERO_EM = 8 # non zero error message (stderr is different)
STDOUT_ERROR_MESSAGE = 9
STDERR_ONLY_DIFF = 10
DMESG_STDERR_ONLY_DIFF = 11
DMESG_STDOUT_ONLY_DIFF = 12
DMESG_OUT_DIFF = 13
DMESG_APP_SPECIFIC_CHECK_FAIL= 14
DMESG_XID_43 = 15

# SDCs
STDOUT_ONLY_DIFF = 16
OUT_DIFF = 17
APP_SPECIFIC_CHECK_FAIL= 18

OTHERS = 19
NUM_CATS = 20

CAT_STR = ["Masked: Error was never read", "Masked: Write before read",
"Masked: other reasons", "DUE: Timeout", "DUE: Non Zero Exit Status", 
"Pot DUE: Masked but Kernel Error", "Pot DUE: SDC but Kernel Error", 
"Pot DUE: Different Error Message", "Pot DUE: Error Message in Standard Output", 
"Pot DUE: Stderr is different", "Pot DUE:Stderr is different, but dmesg recorded", 
"Pot DUE: Standard output is different, but dmesg recorded", 
"Pot DUE: Output file is different, but dmesg recorded", 
"Pot DUE: App specific check failed, but dmesg recorded",
"Pot DUE: Xid 43 recorded in dmesg",
"SDC: Standard output is different", "SDC: Output file is different", 
"SDC: App specific check failed", "Uncategorized"]


#########################################################################
# Error model: Plese refer to the NVBITFI user guide to see a description of 
# where and what errors NVBITFI can inject for the two modes (register file 
# and instruction output-level injections). 
# Acronyms: 
#    bfm: bit-flip model
#    igid: instruction group ID
#########################################################################

# Used for instruction output-level value injection runs 
# G_GPPR and G_GP should be equivalent because we do not inject into predicate regiters in this release. 
inst_value_igid_bfm_map = {
	#G_GP: [FLIP_SINGLE_BIT],
	G_FP32: [FLIP_SINGLE_BIT]#G_LD: [FLIP_SINGLE_BIT]#G_LD: [FLIP_SINGLE_BIT]
#  Supported models
# 	G_GP: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE]
# 	G_FP64: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE]
# 	G_FP32: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE]
# 	G_LD: [FLIP_SINGLE_BIT, FLIP_TWO_BITS, RANDOM_VALUE, ZERO_VALUE] 

}


#########################################################################
# List of apps 
# app_name: [
#			workload directory, 
#			binary name, 
#			path to the binary file, 
#			expected runtime in secs on the target PC, 
#			additional parameters to pass to run.sh script (usually this should be "")
#		]
# run.sh script should be in the workload directory
# golden output files should also be in the workload directory
#########################################################################
"""
apps = {
	#'simple_add': [
	#		NVBITFI_HOME + '/test-apps/simple_add', # workload directory
	#		'simple_add', # binary name
	#		NVBITFI_HOME + '/test-apps/simple_add/', # path to the binary file
	#		1, # expected runtime
	#		"" # additional parameters to the run.sh
	#	],
	'NS_backprop': [
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/backprop', # workload directory
			'backprop_cuda', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/backprop', # path to the binary file
			2.5, # expected runtime
			"65536" # additional parameters to the run.sh
		],
	'NS_NN': [
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/nn', # workload directory
			'nn', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/nn', # path to the binary file
			1.5, # expected runtime
			NVBITFI_HOME +"/test-apps/rodinia_no_SFU/data/nn/list640k_64.txt -r 100 -lat 30.0 -lng 90.0" # additional parameters to the run.sh
		],
	'NS_gaussian': [
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/gaussian', # workload directory
			'gaussian', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/gaussian', # path to the binary file
			2.5 ,# expected runtime
			"-f "+NVBITFI_HOME +"/test-apps/rodinia_no_SFU/data/gaussian/matrix1024.txt" # additional parameters to the run.sh
		],
	'NS_cfd': [
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/cfd', # workload directory
			'euler3d', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/cfd', # path to the binary file
			3.5, # expected runtime
			NVBITFI_HOME +"/test-apps/rodinia_no_SFU/data/cfd/fvcorr.domn.097K" # additional parameters to the run.sh
		],
	'NS_heartwall': [
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/heartwall', # workload directory
			'heartwall', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/heartwall', # path to the binary file
			1, # expected runtime
			NVBITFI_HOME +"/test-apps/rodinia_no_SFU/data/heartwall/test.avi 1" # additional parameters to the run.sh
		],
	'NS_lavaMD': [
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/lavaMD', # workload directory
			'lavaMD', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/lavaMD', # path to the binary file
			4, # expected runtime
			"-boxes1d 4" # additional parameters to the run.sh
		],
	'NS_myocyte': [
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/myocyte', # workload directory
			'myocyte', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/myocyte', # path to the binary file
			2, # expected runtime
			"100 1 0" # additional parameters to the run.sh
		],
	'NS_lud': [
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/lud', # workload directory
			'lud', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/lud', # path to the binary file
			2, # expected runtime
			"-s 256 -v" # additional parameters to the run.sh
		],
	'NS_srad_v1': [
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/srad_v1', # workload directory
			'srad', # binary name
			NVBITFI_HOME + '/test-apps/rodinia_no_SFU/cuda/srad_v1', # path to the binary file
			2, # expected runtime
			"100 0.5 502 458" # additional parameters to the run.sh
		],
}
"""
apps = {
#	'simple_add': [
#			NVBITFI_HOME + '/test-apps/simple_add', # workload directory
#			'simple_add', # binary name
#			NVBITFI_HOME + '/test-apps/simple_add/', # path to the binary file
#			1, # expected runtime
#			"" # additional parameters to the run.sh
#		],
#	'odd_even_sort': [
#			NVBITFI_HOME + '/test-apps/odd_even_sort', # workload directory
#			'odd_even_sort', # binary name
#			NVBITFI_HOME + '/test-apps/odd_even_sort/', # path to the binary file
#			1, # expected runtime
#			"" # additional parameters to the run.sh
#		],
#	'matrix_multiplication': [
#			NVBITFI_HOME + '/test-apps/matrix_multiplication', # workload directory
#			'matrix_multiplication', # binary name
#			NVBITFI_HOME + '/test-apps/matrix_multiplication/', # path to the binary file
#			10, # expected runtime
#			"" # additional parameters to the run.sh
#		],
#	'matmul': [
#			NVBITFI_HOME + '/test-apps/matmul', # workload directory
#			'simple_add', # binary name
#			NVBITFI_HOME + '/test-apps/matmul/', # path to the binary file
#			2, # expected runtime
#			"" # additional parameters to the run.sh
#		],
#	'matmul_float': [
#			NVBITFI_HOME + '/test-apps/matmul_float', # workload directory
#			'simple_add', # binary name--cuda 0 --image IndianPines --checkpoint models/ip/10_IP.pth --model li --pca 10
#			NVBITFI_HOME + '/test-apps/matmul_float/', # path to the binary file
#			5, # expected runtime
#			"" # additional parameters to the run.sh
#		],
	'pca_hyperspectral': [
			NVBITFI_HOME + '/test-apps/PCAHyperspectralClassifier', # workload directory
			'pca.so', # binary name
			NVBITFI_HOME + '/test-apps/PCAHyperspectralClassifier/', # path to the binary file
			20, # expected runtime
			"" # additional parameters to the run.sh
		],
}

#########################################################################
# Separate list of apps and error models for parsing because one may want to
# parse results for a differt set of applications and error models 
#########################################################################
parse_inst_value_igid_bfm_map = inst_value_igid_bfm_map
parse_apps = apps

#########################################################################
# Set paths for application binary, run script, etc. 
#########################################################################
app_log_dir = {} 
script_dir = {} 
bin_dir = {}
app_dir = {}
app_data_dir = {}
def set_paths(): 
	merged_apps = apps # merge the two dictionaries 
	merged_apps.update(parse_apps) 
	
	for app in merged_apps:
		app_log_dir[app] = NVBITFI_HOME + "/logs/" + app + "/"
		bin_dir[app] = merged_apps[app][2]
		app_dir[app] = merged_apps[app][0]
		script_dir[app] = merged_apps[app][0]
		app_data_dir[app] = merged_apps[app][0]

set_paths()

#########################################################################
# Parameterizing file names
#########################################################################
run_script = "run.sh"
nvbit_profile_log = "nvbitfi-igprofile.txt"
injection_seeds = "nvbitfi-injection-info.txt"
inj_run_log = "nvbitfi-injection-log-temp.txt"
stdout_file = "stdout.txt"
stderr_file = "stderr.txt"
output_diff_log = "diff.log"
stdout_diff_log = "stdout_diff.log"
stderr_diff_log = "stderr_diff.log"
special_sdc_check_log = "special_check.log"

#########################################################################
# Number of gpus to use for error injection runs
#########################################################################
NUM_GPUS = 1

use_filelock = False

