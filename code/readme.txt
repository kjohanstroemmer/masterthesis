Dust in the Intergalactic Medium & Type Ia Supernovae
Master thesis project
Karl Johanströmmer
k.johstr@gmail.com
Stockholm University
March 20 2019
--------------------

The main file is digm.py which contains all calculations.
To use it, scroll to the bottom and uncomment whichever case is to be ran.
It is used to produce chains of numbers generated with the MultiNest algorithm.
These chains are placed in the folder chains.
The file chains/1-post_equal_weights.dat contains the posterior distributions of the parameters from the specific run after the sampler has converged.

The file results.py can be used to quickly examine the result of a sampling run.
The files in the chains foler must be removed if another case is to be tested, rename 1-post_equal_weights.dat to something fitting and place it elsewhere.

The results of every sampling are already stored in the folder posteriors.
They are sorted by data set used, which cosmology is fitted and whether a CMB prior is included. 0 for no and 1 for yes.
The file posterior_plots.py can be used to examine these chains. To use it, uncomment whichever case that is to be shown, both in loading of the file and further down in plotting. 
