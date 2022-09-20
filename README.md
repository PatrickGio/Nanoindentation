# Nanoindentation
Utilize machine learning for accelerated parameter identification from indentation data. Machine learing models are trained with synthetic data generated by nonlinear mixed finite element models.




## Quick Start

### Example

<img src="Figures/Figure_5.png" alt="\textbf{Summary of the modelling approaches.}" width="100%"/>

\textbf{Prediction of averaged experimental data: mouse brain slices and chemically fixed brain slices.} Comparison of the averaged loading curves (686 samples) from mouse brain slices (black triangle marker), and the averaged loading curves (686 samples) from chemically fixed mouse brain slices (black circle marker) were plotted with the neural network predictions in red and grey, respectively. (A) Neo-Hookean material model with the least squares ML approach, (B) Gent material model with the least squares ML approach, (C) Neo-Hookean material model with the direct inverse ML approach, and (D) Gent material model with the direct inverse ML approach. Predicted material parameters are included in the respective legend.





## Summary

### Synthetic Data Generation

<img src="Figures/Figure_1.png" alt="\textbf{Summary of the modelling approaches.}" width="100%"/>

**Summary of the modelling approaches.** A) LHS was used to sample the four parameter input space ( $\delta y$, $W$, $H$, and $\mu$) for the neo-Hookean material model, and five parameter input space ( $\delta y$, $W$, $H$, $\mu$, and $Jm$) for the Gent material model to generate a FE input file. B) The FE input file was fed into the implicit mixed FE model (C) to generate a load-displacement curve output, FE output file. (A-C) represents the forward problem, while the inverse problem, determining material parameters from experimental data, is accomplished through the use of two machine learning models. 


### Inverse Problem

<img src="Figures/Figure_2.png" alt="\textbf{Summary of the modelling approaches.}" width="100%"/>


(D) the first machine learning model used a neural network to learn the forward problem, predict the loading curve ( $P^*_n$) from material properties ( $\mu^*$, $Jm^*$) and sample dimensions ( $W^*$, $H^*$), which is called as the mapping function for a nonlinear least squares algorithm to solve the inverse problem. (E) the second machine learning model used a neural network to directly learn the inverse problem, predict material parameters ( $\mu$, $Jm$) from sample dimensions ( $W^*$, $H^*$), loading curve ( $P^*_n$), and the slope of the loading curve ( $S^*_n$)




### Machine Learning 

<img src="Figures/Figure_4.png" alt="\textbf{Summary of the modelling approaches.}" width="100%"/>

\textbf{Model predictions of synthetic data.} A) Comparison of the neural network (black dots) prediction of unseen data to the Hertzian solution (red triangle) and Modified Hertzian solution (orange squares). Predicted shear modulus is plotted against target shear modulus, where the dotted red line is a perfect prediction. B) Magnification of A. C) Comparison of experimental data with 0.1R max indentation to neural network prediction. D) Comparison of experimental data with 0.5R max indentation to neural network prediction.}



### Experimental Data: Brain Tissue




<img src="Figures/Figure_6.png" alt="\textbf{Summary of the modelling approaches.}" width="100%"/>

{\textbf{Prediction of experimental data: mouse brain slices and chemically fixed brain slices. } The predicted material parameters for the Gent material model with the direct inverse ML approach were plotted for the mouse brain slices (A,C) (grey circle markers) and chemically fixed mouse brain slices (B,D) (red circle markers). The Gent material parameters for the averaged experimental loading curves for the mouse brain slices (red dotted line) and chemically fixed brain slices (black dotted line) are plotted over the parameters determined for the individual data samples.}



## Layout of Repository

### SynthData
### Trained_NN_Models
### Functions
### ExperimentalData
### Results



