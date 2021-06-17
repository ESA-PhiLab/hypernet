************
Installation
************

Core install
-----------------

1) Make sure you have Python 3.6 installed. It can be downloaded and installed from https://www.python.org/downloads/
2) Make sure you have conda installed. Full guide for your operating system is available here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
3) Copy the repo https://github.com/ESA-PhiLab/hypernet onto your computer and enter the `hypernet/beetles` directory.
4) To create the environment with all the required dependencies, execute the following line:

    .. code-block:: python

        conda env create -f environment.yml
5) Activate the environment by running:

    .. code-block:: python

        conda activate decent

Standard Tensorflow install
---------------------------
If you are not interested in the model quantization and compilation, you can proceed with the following steps. However, if you are,
please move to the next section.


If you are going to run your code on CPU, execute:

.. code-block:: python

    pip install tensorflow==1.12

or for the GPU:

.. code-block:: python

    pip install tensorflow-gpu==1.12

Xillinx toolbox install
-----------------------

The Xillinx toolbox is required to run the model quantization and compilation. As of now, it is only required for the `model_quantization_xillinx` example. So if you are not interested in that,
you can skip this part.

1) For this part, you need **Ubuntu** 14.04, 16.04 or 18.04. 
2) Download the `xillinx_dnndk_v3.1_190809.tar.gz` from https://www.xilinx.com/products/design-tools/ai-inference/ai-developer-hub.html#edge
3) Execute 

    .. code-block:: python

        pip install ${TF_DNNDK_WHEEL_PATH}
Where `TF_DNNDK_WHEEL_PATH` is a path to the tensorflow `.whl` package located in the directory you have just downloaded. Select the one that suits you, based on the version of the operating system,
python version and whether you would like to run on CPU or GPU. Keep in mind that you will need CUDA and cuDNN libraries for the GPU. If you run into any problems, the more detailed installation process could be found here: https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf

4) To validate the installation, run:

    .. code-block:: python

        decent_q --help



If you have completed the setup correctly, you are now able to run the code and the examples. The process of installing and running the jupyter notebooks is described in the official documentation: https://jupyter.readthedocs.io/en/latest/install.html