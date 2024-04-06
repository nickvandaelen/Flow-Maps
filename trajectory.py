import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce
from copy import deepcopy
import os
import json

getDictKeys = lambda dict_: list(dict_.keys())
getDictValues = lambda dict_: list(dict_.values())
getDictItems = lambda dict_: list(dict_.items())
euclDist = lambda l1, l2: sum([(l1[i] - l2[i])**2 for i in range(len(l1))])**0.5

def write_to_json(data, filename):
    """
    Write a dictionary to a JSON file.
    
    Args:
        data: Dictionary to be written to the file.
        filename: Name of the JSON file.
    """
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def read_from_json(filename):
    """
    Read a JSON file and return its content as a dictionary.
    
    Args:
        filename: Name of the JSON file.
        
    Returns:
        Dictionary containing the content of the JSON file.
    """
    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    return data

def min_max_scaling(data, min_val=0, max_val=1, axis=0):
    """
    Perform min-max scaling on the given data, scaling it to a custom range.
    
    Args:
        data: A NumPy array or list containing the data to be scaled.
        min_val: The minimum value of the custom range (default is 0).
        max_val: The maximum value of the custom range (default is 1).
        axis: Axis or axes along which to operate (default is 0).
        
    Returns:
        Scaled data.
    """
    # Convert data to NumPy array if it's not already
    data = np.array(data)
    
    # Calculate the minimum and maximum values for each feature
    min_vals = np.min(data, axis=axis)
    max_vals = np.max(data, axis=axis)
    
    # Perform min-max scaling to the custom range
    scaled_data = min_val + (data - min_vals) * (max_val - min_val) / (max_vals - min_vals)
    
    return scaled_data


#### Quick Sort Function ####
# Function to find the partition position
def partition(array, low, high):

	# choose the rightmost element as pivot
	pivot = array[high]

	# pointer for greater element
	i = low - 1

	# traverse through all elements
	# compare each element with pivot
	for j in range(low, high):
		if array[j][1] <= pivot[1]:

			# If element smaller than pivot is found
			# swap it with the greater element pointed by i
			i = i + 1

			# Swapping element at i with element at j
			(array[i], array[j]) = (array[j], array[i])

	# Swap the pivot element with the greater element specified by i
	(array[i + 1], array[high]) = (array[high], array[i + 1])

	# Return the position from where partition is done
	return i + 1

# function to perform quicksort


def quickSort(array, low, high):
	if low < high:

		# Find pivot element such that
		# element smaller than pivot are on the left
		# element greater than pivot are on the right
		pi = partition(array, low, high)

		# Recursive call on the left of pivot
		quickSort(array, low, pi - 1)

		# Recursive call on the right of pivot
		quickSort(array, pi + 1, high)

class PointError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class Trajectory:
    """Class to read text file containing trajectories and perform some filter operations on those trajectories."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.trajectories_str = [] # Assign list for trajectories as strings (in text file)
        self.trajectories = [] # Assign list for trajectories as lists
        self.read() # Read trajectory text file
        self.unique_points = self.getUniquePoints()
        self.unique_trajectories = self.getUniqueTrajectories()

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, index):
        return self.trajectories[index]
    
    def __contains__(self, item):
        return item in self.trajectories
    
    def append(self, item):
        self.trajectories.append(item)
    
    def extend(self, items):
        self.trajectories.extend(items)

    def getTrajectories(self):
        return self.trajectories
    
    def getTrajectoriesStr(self):
        return self.trajectories_str
    
    def getUniqueTrajectories(self):
        """Returns the unique trajectories in the list of trajectories."""
        unique_trajectories = []
        for trajectory in self.trajectories:
            if trajectory not in unique_trajectories:
                unique_trajectories.append(trajectory)
        return unique_trajectories

    def read(self):
        """Reads the trajectories from the text file and stores them."""
        # Open the file in read mode
        with open(self.file_path, 'r') as file:
            # Read each line of the file using a loop
            for line in file:
                # Process each line as needed
                newline = line.strip()  # Remove newline character
                self.trajectories_str.append(newline) # Add to string list
                self.trajectories.append(newline.split()) # Add to nested list

    def describe(self):
        """Returns some useful information about the trajectories."""
        # Print nicely
        for info, nr in zip(["Total", "Unique", "# points"], [len(self.trajectories), len(self.unique_trajectories),
                                                              len(self.unique_points)]):
            print(f"{info}", end=": ")
            print(f"{nr}")
            print("_"*15)
    
    def getUniquePoints(self):
        """Returns unique points in the trajectories."""
        # Flatten the nested list into a single list
        flat_list = [item for sublist in self.trajectories for item in sublist]
        # Get unique elements using set
        unique_elements = set(flat_list)
        # Convert the unique elements back to a list if needed
        unique_elements_list = sorted(list(unique_elements))
        return unique_elements_list
    
    def filterOnStartPoint(self, startPoint):
        """Filter the trajectories on a given start node and return trajectories having that node as a starting point."""
        if startPoint not in self.unique_points:
            raise PointError(f"Point {startPoint} not found!")
        # Filter trajectories based on starting point
        filtered_trajectories = [trajectory for trajectory in self.trajectories if trajectory[0] == startPoint]
        return filtered_trajectories
    
    def filterOnStartPoints(self, startPoints):
        """Filter the trajectories on given start nodes and return trajectories having one of these nodes as a starting point."""
        for point in startPoints:
            if point not in self.unique_points:
                raise PointError(f"Point {point} not found!")
        # Filter trajectories based on starting point
        filtered_trajectories = []
        for point in startPoints:
            filtered_trajectories.extend(self.filterOnStartPoint(point))
        return filtered_trajectories
    
    def filterOnEndPoint(self, endPoint):
        """Filter the trajectories on a given end node and return trajectories having that node as an ending point."""
        if endPoint not in self.unique_points:
            raise PointError(f"Point {endPoint} not found!")
        # Filter trajectories based on ending point
        filtered_trajectories = [trajectory for trajectory in self.trajectories if trajectory[-1] == endPoint]
        return filtered_trajectories
    
    def filterOnEndPoints(self, endPoints):
        """Filter the trajectories on given end nodes and return trajectories having one of these nodes as an ending point."""
        for point in endPoints:
            if point not in self.unique_points:
                raise PointError(f"Point {point} not found!")
        # Filter trajectories based on ending point
        filtered_trajectories = []
        for point in endPoints:
            filtered_trajectories.extend(self.filterOnEndPoint(point))
        return filtered_trajectories
    
    def filterOnIntermediatePoint(self, point, includeStart=False, includeEnd=False):
        """Filter the trajectories on a given intermediate point and return trajectories having that node as a traversing point."""
        if point not in self.unique_points:
            raise PointError(f"Point {point} not found!")
        # Filter trajectories based on intermediate point
        filtered_trajectories = [trajectory for trajectory in self.trajectories if point in trajectory]
        # Remove point if it is starting point (if specified)
        if not(includeStart):
            filtered_trajectories = [trajectory for trajectory in filtered_trajectories if trajectory[0] != point]
        # Remove point if it is ending point (if specified)
        if not(includeEnd):
            filtered_trajectories = [trajectory for trajectory in filtered_trajectories if trajectory[-1] != point]
        return filtered_trajectories
    
    def filterOnIntermediatePoints(self, points, includeStart=False, includeEnd=False):
        """Filter the trajectories on given intermediate nodes and return trajectories having one of these nodes as a traversing point."""
        for point in points:
            if point not in self.unique_points:
                raise PointError(f"Point {point} not found!")
        # Filter trajectories based on intermediate point
        filtered_trajectories = []
        for point in points:
            filtered_trajectories.extend(self.filterOnIntermediatePoint(point, includeStart, includeEnd))
        return filtered_trajectories
    
    def filterOnLength(self, length):
        """Filter the trajectories on a given number of nodes and return trajectories having that length."""
        if length < 2:
            raise ValueError("Trajectories must have at least length 2!")
        # Filter trajectories based on length
        filtered_trajectories = [trajectory for trajectory in self.trajectories if len(trajectory) == length]
        return filtered_trajectories
    
    def filterOnLengths(self, lengths):
        "Filter the trajectories on given lengths and return trajectories having one of these lengths."
        for length in lengths:
            if length < 2:
                raise ValueError("Trajectories must have at least length 2!")
        # Filter trajectories based on lengths
        filtered_trajectories = []
        for length in lengths:
            filtered_trajectories.extend(self.filterOnLength(length))
        return filtered_trajectories
    
    def filterOnStartAndEndPoints(self, startPoints, endPoints):
        """Filter the trajectories on given start and end nodes and return trajectories having these nodes as start or end points."""

        # Raise errors if inputs are not correct
        for startPoint, endPoint in zip(startPoints, endPoints):
            if startPoint not in self.unique_points: # Start point not in points
                raise PointError(f"Point {startPoint} not found!")
            if endPoint not in self.unique_points: # End point not in points
                raise PointError(f"Point {endPoint} not found!")
            if startPoint in endPoints: # Start point is in end points (illegal)
                raise PointError(f"Start point {startPoint} cannot be in end points!")
            if endPoint in startPoints: # End point is in start points (illegal)
                raise PointError(f"End point {endPoint} cannot be in start points!")
            
        # Filter trajectories based on start & end points
        filtered_trajectories = []
        for trajectory in self.trajectories:
            if (trajectory[0] in startPoints) or (trajectory[-1] in endPoints):
                filtered_trajectories.append(trajectory)
        return filtered_trajectories
    
    def countTrajectories(self, reverse=True, relative=False):
        """Returns a dictionary containing the count of each single trajectory in the trajectories."""
        # Initialize an empty dictionary to store counts
        element_counts = {}

        # Count occurrences of each element
        for element in self.trajectories_str:
            if element in element_counts:
                element_counts[element] += 1
            else:
                element_counts[element] = 1
        
        # Transform into tuple pairs
        element_counts = getDictItems(element_counts)
        element_counts = sorted(element_counts, key=lambda x: x[1], reverse=reverse)

        # Transform again into dictionary
        element_counts = dict(element_counts)

        # Get relative counts if specified
        if relative:
            total = sum(getDictValues(element_counts))
            for key in getDictKeys(element_counts):
                element_counts[key] = element_counts[key] / total
        
        return element_counts
    
    def constructWeightMatrix(self, trajectories=None, as_df=False):
        """Construct a matrix of edge weights from the list of individual trajectories."""
        # If trajectories is None, then select all trajectories
        if trajectories is None:
            trajectories = self.trajectories
        
        # Initialize matrix with zeros
        matrix = np.zeros((len(self.unique_points), len(self.unique_points)))

        # Make dataframe to easily add weights
        df_matrix = pd.DataFrame(matrix, columns=self.unique_points, index=self.unique_points)

        # Add 1 for each existing double in trajectory
        for trajectory in trajectories:
            for i in range(len(trajectory)-1):
                m, n = trajectory[i:i+2]
                df_matrix.loc[m, n] += 1
        
        if as_df:
            return df_matrix
        return df_matrix.to_numpy()