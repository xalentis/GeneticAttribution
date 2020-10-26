# GeneticAttribution
Lab of Origin prediction of DNA sequences

Implementation of paper: Deep learning to predict the lab-of-origin of engineered DNA

Nielsen, A and Voigt, C (Voigt Labs)

URL: https://www.nature.com/articles/s41467-018-05378-z.pdf?origin=ppub

Paper results in predictive accuracy of 48%, this model delivers 75.4% on validation set (10%)


NOTE: The dataset used here provided from this URL:
https://www.drivendata.org/competitions/63/genetic-engineering-attribution
60,000 DNA sequences with lab of origin as label


#### WARNING: This model creates a 60391x16048x4 matrix requiring at least 30GB RAM, 64GB recommended minimum
Training done on Intel i9 10-core with dual Nvidia RTX2080rti GPU stack

Training time: 157 mins for 25 epochs


Showcases some good examples of using R to approach a problem:
- Optimize routines by packaging as C++ code
- One-hot encoding of very large matrices
- Finding optimal neural-network parameters
- Loading/Saving objects to disk
- Class weight calculation for unbalanced datasets
- Using customer Keras data generators in R
- Custom scoring
