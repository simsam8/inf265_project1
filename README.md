# INF265 Project1

## Setup

The project was created with Python 3.11. To run and replicate our results, make sure to install the project dependencies:
`pip install -r requirements.txt`

To view and run the notebooks, launch Jupyter Notebook with `jupyter notebook` and select the .ipynb files to open them.

To reproduce our work with identical results, you should set the seed for the randoom state to 265, use CUDA (train on GPU) and set the default Pytorch data type to double. 

## File Structure

- [docs](docs): Project assignment description and checklist.
- [imgs](imgs): Folder containing output images from notebooks.
- [report.md](report.md): Markdown for generating the project report.
- [report.pdf](report.pdf): The project report. Explains the design choices and general approach for solving the assignment. 
- [requirements.txt](requirements.txt): List of the Python dependencies for this project. 
- [backpropagation.ipynb](backpropagation.ipynb): Notebook implementing backpropagation. 
- [gradient_descent.ipynb](gradient_descent.ipynb): Notebook implementing gradient descent, training loop, model evaluation and selection
- [tests_backpropagation.py](tests_backpropagation.py): Tests to verify our implementation of backpropagation. Not written by us. 
