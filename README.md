
This repo evaluate weight pruning efficency (see Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding paper) when combined with VCL technique (see E. Littwin, L. Wolf. Regularizing by the Variance of the Activations' Sample-Variances. Neural Information Processing Systems (NIPS) paper).


The main file is run_me.py. Run this file to train and test the model with different configurations (model depth, activation type,
activation regularizer and which dataset fold to use).

Results are saved in the result directory, each time run_me.py is called a new dir is created inside /result dir with 
a name matching the current timestamp. Inside /result/timestamp/ you will find:
  1. A different directory for each model configuration, each one of those directories contains:
  
    1.1 A .log file detailing the training process
    1.2 A .csv file detailing the results
    1.3 A .json file detailing the specific configuration. This file can be used to reproduce the results.
    
  2. A .csv file containing a result summary for all configurations.
 

At the moment the code supports image-segmentation dataset (UCI). The dataset itself can be found inside ./datasets folder.

