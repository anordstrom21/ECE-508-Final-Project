# ECE-508-Final-Project
Repo for ECE 508 Final Project

The project assumed that you already have a Python 3 or above installed on your machine.

### To run the model

In order to use the model we have already trained, executing the `use_model.py` will prompt you for three inputs, the prompt should be self-explanatory.

### To train the model

To actually train a model with the code we are providing, you can run the `trainer.py`; however, assuming that you don't have a GPU that Pytorch is able to utilize as a compute resource, we are also providing Jupyter notebook file you can run on cloud resources like Google Colab (this is the one we used to make ShakGPT_3_512).

In `trainer.py` file, starting from the line 123, you can edit the name of the model that will be produced by the `trainer.py`, and the amount of epochs `trainer.py` will take to create the model.