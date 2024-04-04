import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce
from copy import deepcopy
import os
print(os.getcwd())

getDictKeys = lambda dict_: list(dict_.keys())
getDictValues = lambda dict_: list(dict_.values())
getDictItems = lambda dict_: list(dict_.items())
euclDist = lambda l1, l2: sum([(l1[i] - l2[i])**2 for i in range(len(l1))])**0.5

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

# Get trajectories
t = Trajectory(r"algorithm_repository\Flow-Maps\trajectories.txt")
print(t.countTrajectories(relative=True))