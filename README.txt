
MATLAB Code for the paper: Online Passive-Aggressive Multilabel Classification Algorithms 

T. Zhai and H. Wang, "Online Passive-Aggressive Multilabel Classification Algorithms," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2022.3164906.

## guideline for how to use this code


1. "run_PAML.m" is a demo for the linear PAML algorithm proposed in this paper, which relies on  "PAML_sparse.c" to run.
You should first input "mex -largeArrayDims PAML_sparse.c" in the command window of matlab in order to build a executable mex-file.
Then run "run_PAML.m".
   "run_PAML_I.m" is a demo for the linear PAML-I algorithm. "run_PAML_II.m" is a demo for the linear PAML-II algorithm.
Both files have similar running way as that for  "run_PAML.m".


2. "kernel_PAML.m" is a demo for the kernelized PAML algorithm.
Before running this program, the kernel matrix has been precalculated for accelerating the computing. 
So if you want to change the dataset, please follow the steps below to run the program:
(1) run "precalculate_kernelMatrix.m" to create the kernel matrix
(2) run "kernel_PAML.m".

"kernel_PAML_I.m" is a demo for the kernelized PAML-I algorithm.
"kernel_PAML_II.m" is a demo for the kernelized PAML-II algorithm.
Both files have similar running way as that for  "kernel_PAML.m".


ATTN: 
- This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Tingting ZHAI (zhtt@yzu.edu.cn).
- This package was developed by Tingting ZHAI (zhtt@yzu.edu.cn). For any problem concerning the code, please feel free to contact Mrs.ZHAI.