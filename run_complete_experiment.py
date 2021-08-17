import os

from results_analysis.results_analysis import analyze
from run_all_classes_mnist import run_all_mnist
from run_all_classes_cifar import run_all_cifar
from FL_neuron_subset import intersections

run_all_mnist(8)  # this value is the random seed, use 8 to reproduce the results of the paper,
run_all_cifar(8)  # change it to see similar but different results

intersections("mnist")
intersections("cifar10")

os.chdir("results_analysis")
analyze("mnist")
analyze("cifar10")


