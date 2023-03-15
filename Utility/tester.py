from re import T
import numpy as np
import matplotlib.pyplot as plt

import utils

utils.DEBUG = True

size=[100,100]

feats1=[(10,5),(4,17),(51,51),(99,99)]
feats2=[(5,7),(51,10),(55,56),(60,75)]
feats3=[(30,41),(12,68),(80,51),(55,60)]


num_levels=3

print('Score: ', utils.cell_variance(feats1, feats2, size, num_levels))

print('Score: ', utils.cell_variance(feats1, feats3, size, num_levels))
