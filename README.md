# Colorizing images using Deep Neural Networks
Small project testing colorization of black-white/greyscaled images using some neural networks.  
Takes the simple bw version of the images as base and tries to generate the original gt.  

`Train_with_demo.ipynb` contains the main code, the other notebook were created for testing purposes.

Currently the model used is an UNET model using effnet-b2. Other architectures might perform better, especially GANs (as other projects have shown).

Here's a small example gif how the images of the test set can change per epoch.
![](bw_to_color.gif)  

There are some obvious problem atm and various possible improvements that can be implemented. I will test them if I find time and motivation.