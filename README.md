# TTF-Brain-Visualization
###  | [Full Thesis](https://doi.org/10.11575/PRISM/43331) | [Paper](https://doi.org/10.1016/j.cag.2024.104067) |
Code release for "Transferring Transfer Function (TTF): A Guided Approach to Transfer Function Optimization in Volume Visualization" (Brain Visualization). This project transfers a transfer function from one MRI brain volume to another brain volume. This project used [Calgary-Campinas dataset](https://sites.google.com/view/calgary-campinas-dataset/home).

## Concept
Visualizing volumetric data is a key tool for effective communication, with the transfer function playing a pivotal role in deciding what aspects of the volume to display and how.

https://github.com/user-attachments/assets/823b8f6f-cc1c-4822-9101-dab9d6a4955e

However, in MRI volumes, the data values can vary significantly due to factors like manufacturer differences and patient-specific details. As a result, a transfer function designed for one volume may not be effective for another. In this project, our TTF method is employed to optimize and visualize brain MRIs based on examples. By providing a reference volume and its transfer function, the method can visualize new, unseen MRI volumes.


https://github.com/user-attachments/assets/0d749412-78ab-4e57-9574-baa59ff959df




## System Specification
The code was tested on:

* OS: Ubuntu 22.04.4 LTS
* GPU: NVIDIA RTX A6000 with 48 GB Memory
* CPU: AMD Ryzen 9 7950X
* RAM: 128 GB System Memory

## How to run the code
Download or clone this repo.
After installing Anaconda, you can find all the requirements in the environment.yml file. You can create a new environment with all the requirements using,
```
conda env create -f environment.yml
```
After activating the environment, you can run the default script using Python.
```
cd script/ref-GE-target-philips
python main.py
```

**Note:** The batch size was set based on our available GPU memory. You can change the batch size depending on the GPU memory you have available.
