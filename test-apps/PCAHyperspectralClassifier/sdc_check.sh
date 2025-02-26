#!/bin/bash
#
# Copyright 2020, NVIDIA CORPORATION.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

##################################################################################
# Brief explanation of the log files:
# (1) diff.log: If the program generates a separate output file (e.g.,
# output.txt), diff the file with the golden output file and store the diff in
# diff.log. The program output should be deterministic. Please exclude
# non-deterministic values  (e.g., runtimes) from the file before you use
# `diff` or similar utilities.
# (2) stdout_diff.log: diff stdout generated by the program with # the golden
# stdout and store it in stdout_diff.log. Remove the string that matches
# ":::Injecting.*:::" from the stdout.
# (3) stderr_diff.log: diff stderr generated by the program with the golden
# stderr and store it in stderr_diff.log.  
# (4) special_check.log: if the application specific check fails, the
# special_check.log should contain some non-empty string.
#
# If the user prefers to skip one of these checks, he/she should create an
# empty log file (e.g., touch diff.log). 
##################################################################################

# if your program creates an output file (e.g., output.txt) compare it to the file created just now and store the difference in diff.log
# Example: diff output.txt ${APP_DIR}/golden_output.txt > diff.log
#touch diff.log 

#cp ${APP_DIR}/prediction_inference.tif ./
#mv ${APP_DIR}/prediction_inference.tif ./

# comparing prediction image generated by your program
#diff stdout.txt ${APP_DIR}/golden_stdout.txt > stdout_diff.log
diff prediction_inference.tif ${APP_DIR}/golden_prediction_inference.tif > diff.log

# comparing stderr generated by your program
tr '\r' '\n' <stderr.txt | grep -v Inference | grep -v Downloading >stderr_clear.txt
tr '\r' '\n' <${APP_DIR}/golden_stderr.txt | grep -v Inference | grep -v Downloading >golden_stderr_clear.txt
diff -B -w stderr_clear.txt golden_stderr_clear.txt > stderr_diff.log

# comparing stdout generated by your program
grep -v '[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9][0-9]' stdout.txt | grep -v 'Time for' >stdout_clear.txt
grep -v '[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9][0-9]' ${APP_DIR}/golden_stdout.txt | grep -v 'Time for' >golden_stdout_clear.txt
diff -B -w stdout_clear.txt golden_stdout_clear.txt > stdout_diff.log

# Application specific output: The following check will be performed only if at least one of diff.log, stdout_diff.log, and stderr_diff.log is different

# IMPORTANT: WRITE SOMETHING IN "special_check.log" WHEN WHETHER THE OUTPUT IMAGE (BY CHECKING diff.log), THE STDERR OR THE STDOUT(MAYBE, CHECK THIS LATER)
# CHANGE IN ORDER TO TRIGGER NVBITFI TO CHECK WHAT EXACTLY WENT WRONG (E.G. EITHER THE IMAGE OUTPUT, STDOUT OR STDERR ARE DIFFERENT). NVBITFI DOESN'T USE THE
# CONTENT OF "special_check.log" TO ASSESS THIS. THIS FILE IS USED JUST TO LET THE TOOL KNOW ONE OF THESE 3 THINGS WERE AFFECTED BY THE INJECTED FAULT.
cat diff.log > special_check.log
cat stderr_diff.log >> special_check.log
cat stdout_diff.log >> special_check.log

# Delete Datasets
rm -r ./Datasets