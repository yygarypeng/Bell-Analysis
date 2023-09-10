# Bell Study Analysis

These codes are the analysis programs for analyzing the data from PhysicsDAODMaker or the dump root files from the CAF of HWW Analysis Group.
The analysis framework needs to use the specified conda environment.    

p.s. The attached `yaml` file is for the **Mac M-series chips** and the Mt. GPU docker container.    
(Other installations can be found here, https://github.com/jeffheaton/t81_558_deep_learning)
For Apple M-series chips:    
```shell=True
conda env create -f cern_apple.yml
```
For GPU server amd-64 x86:    
```shell=True
conda env create -f cern.yml
```
