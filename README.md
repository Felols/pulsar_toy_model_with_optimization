# SA1114X-KEX
Students: Felix Olsson and Daniel Skott

Supervisors:  Inga Saathoff and Christos Tegkelidis

This repository contains the current iteration of a Pulsar toy-model developed by students Felix Olsson and Daniel Skott with support from Inga Saathoff and Christos Tegkelidis all from KTH.
The work done is a Bachelor Thesis project titled "Pulse Profile Modeling of Accreting X-ray Pulsars: A Case Study of Centaurus X-3" and is available at the [KTH diva portal](https://kth.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=5&af=%5B%5D&searchType=SIMPLE&sortOrder2=title_sort_asc&query=Felix+Olsson&language=sv&pid=diva2%3A1979112&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=3944) .
## Toy model

The `toymodel.py` file will when run open a GUI that allows the user to experiment with different geometrical parameters aswell as different emission patterns.

![toy_model_image](/Image_folder/toymod4cut.png)

## Optimization

There are two optimization methods used one is Particle Swarm Optimziation this one can be found in the `PSOrunner.py` file where the end user may change the PSO variables freely.
The second one is a Markov Chain Monte Carlo simulation that can be found in `MCMCrunner.py`, this file utilizes

### Particle Swarm Optimization

![pso_image](/Image_folder/phaseprof.png)

#### Data storage

The PSO optimizations runs previous paramaters can be found in various .csv files with names depending on what kind of fitting combination is used since there are two data sets for the two poles.

These .csv files include the fit value, wheter or not the different poles have mirrored emission patterns, the emission function, dipole criteria, runtime
and the PSO parameters particle ammount, iterations, inertia weight, cognitive weight and social weight.

The geometry parameters themselvs are in a list in the column parameters in order of inclination, positional angle, magnetic colatitude (offset), phase shift, inclination offset, azimuth offset, radius (as a Schwarzschild ratio) , gamma1, weight1, weight2, gamma2 and phs2.
Given in degrees where they apply. 
The gammas are used for the narrowing the emission pattern, gamma2 is only used for combined emission types and the weights are used for combined functions.

The file `plotter.py` can be used to reconstruct the intensity plots from these .csv files.

### Markov Chain Monte Carlo 

For Markov Chain Monte Carlo the `emcee` library was used also the module `corner` is used to generate corner plots.
Data from the Markov Chain Monte Carlo simulations are not stored in this github

![mcmc_image](/Image_folder/MCMCcorner1.png)
