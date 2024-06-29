#### Import packages ####
import copy
import itertools
import math
import os
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
from matplotlib import cm

import trajectory as traj

#### Get trajectory data ####
trajectory_data = traj.Trajectory(r"algorithm_repository\Flow-Maps\airport_trajectories_NEW.txt")
edge_matrix = trajectory_data.constructWeightMatrix(None, True)