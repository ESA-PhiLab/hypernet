************
Installation
************

For the environment management in our repo, we utilize `conda`.

Required dependencies are stored in the `environment.yml` file. To create the environment with required packages, execute the following command:

.. code-block:: python

    conda env create -f environment.yml


Because of the fact that we utilize the Xillinx DNNDK tool for model quantization and compilation, the `tensorflow` package has to be installed manually. Please refer to the official documentation, where the process is fully described: (https://www.xilinx.com/support/documentation/sw_manuals/ai_inference/v1_6/ug1327-dnndk-user-guide.pdf - Chapter 1: Quick Start - Tensorflow Version: Installing with Anaconda).

Keep in mind, that the DNNDK tool requires **Ubuntu** 14.04, 16.04 or 18.04.