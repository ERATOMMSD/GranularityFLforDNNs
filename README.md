# What to Blame? On the Granularity of Fault Localization for Deep Neural Networks

This repository contains the code to reproduce the experiments of the paper *What to Blame? On the Granularity of Fault Localization for Deep Neural Networks*, published in ISSRE'21.

## Structure of the repository
* Folder *neural_networks* contains the DNNs used in the experiments
* Folder *results_analysis* contains the scripts to produce experimental results
* Folder *experimentalResults* contains the experimental results reported in the paper

## Requirements
To run the scripts of this repository, you need a python environment with several libraries installed, including keras and pingouin. To have a working installation in Linux, you can follow these steps:
* Install Anaconda. You can download Anaconda for your OS from [https://www.anaconda.com/](https://www.anaconda.com/), e.g.,
  ```
  wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
  bash Anaconda3-2020.11-Linux-x86_64.sh
  ```
* Create an environment (e.g., with name "myenv") using the requirements listed in *requirements/requirements_linux64.txt*
  ```
  conda create -n myenv --file requirements/requirements_linux64.txt
  ```

Before running the code, you need to activate the conda environment:
```
conda activate myenv
```

## Reproducing the experiments
In order to reproduce the same experiments reported in the paper, simply execute *run_complete_experiment.py*

## Running DeepFault and DeepFaultGr - Command Line Settings
You can also run the original version of DeepFault or its granular version DeepFaultGr, using these command line arguments:
* model_name: Name of the -Keras- model file. Note that the architecture file (i.e., json) and the weights file should be saved separately. If you already trained and saved your model into one file, you might want to change the corresponding "load_model" function accordingly.
* dataset: Name of the dataset to be used. Current implementation supports only 'mnist' and 'cifar'. However, adding another is not difficult.
* C (selected_true_class): In DeepFault, we find the suspicious neurons for each class separately. This argument is a number between 0 and 9 for MNIST or CIFAR.
* MC (selected_misclassified_class): This argument is a number between 0 and 9 for MNIST or CIFAR. In DeepFaultGr, we also distinguish between what label (class) was predicted for the input. The script uses all inputs that were correctly predicted, or predicted as selected_misclassified_class. Use MC == -1 to execute the non-granular version of DeepFault.
* step_size: We multiply the gradient values by this parameter for scaling up or down the change while synthesizing a new input.
* distance: The maximum amount of distance (l_inf norm) between the original and the synthesized input.
* approach: The approach for finding the suspicious neurons. Current implementation supports 'tarantula', 'ochiai', 'dstar', and 'random'. 
* SN (susp_num): Number of neurons considered suspicious.
* seed: Seed for the random process. Added for reproducibility. Use 8 to reproduce the same results presented in the paper, or use another random number as in random.randint(0, 10).
* star: This corresponds to the "star" parameter of the dstar approach.
* logfile_name: Name of the file where the results will be saved.
* load_selected_images: flag to signal if DeepFault tries to use the same images to perturb as the granular version. If *True*, it can only run after having the selected_images for all 9 MC of that C. If *False*, it simply synthesizes 10 images starting from 10 random images.
* save_synthesized: flag to signal if the synthesized images are saved. *False* by default. Use *True* if later you want to compare the synthesized images vs the originals using csv_image_viewer.

An example of command is as follows:
```
python run.py --model mnist_test_model_5_30_leaky_relu --dataset mnist -SN 5 -C 2 -MC 3 --approach tarantula -S 8
```

## People
* Matias Duran
* Xiao-Yi Zhang https://group-mmm.org/~xiaoyi/
* Paolo Arcaini http://group-mmm.org/~arcaini/
* Fuyuki Ishikawa http://research.nii.ac.jp/~f-ishikawa/en/

## Paper
M. Duran, Xiao-Yi Zhang, P. Arcaini, and F. Ishikawa. What to Blame? On the Granularity of Fault Localization for Deep Neural Networks. In 2021 IEEE 32nd International Symposium on Software Reliability Engineering (ISSRE)

## Copyright Notice
Note that our code is based on and extends DeepFault. See *originalDeepFaultLicense* for the original DeepFault license.
