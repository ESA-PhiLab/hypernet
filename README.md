# HYPERNET

HYPERNET is a library which implements state-of-the-art and new algorithms for (among others):

* accurate hyperspectral image (HSI) segmentation and analysis using deep neural networks,
* optimization of deep neural network architectures for hyperspectral data segmentation,
* hyperspectral data augmentation,
* validation of existent and emerging HSI segmentation algorithms,
* simulation of multispectral data using HSI.

# BEETLES

HYPERNET is a project that is a follow-up of HYPERNET, and expands our battery of algorithms in the following (this list will be updated):

* generating noisy test data by injecting simulated noise of a given distribution (e.g., Gaussian, impulsive, Poisson), 
* quantization and DPU compilation of the deep neural networks, e.g., with the use of the [Xillinx DNNDK](https://www.xilinx.com/products/design-tools/ai-inference/edge-ai-platform.html#dnndk) tool,
* deep learning-powered hyperspectral unmixing.

# Requirements

The main requirements in python 3.6, available from https://www.python.org/downloads/

GUI application uses QT5, it can be downloaded from https://www.qt.io/download-qt-installer

All other requirements are listed in `requirements.txt` and they can be installed by running `pip install -r requirements.txt`
