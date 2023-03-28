import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy import stats


class PointCloud:
    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32, sep=' ')
            data = data.reshape((-1, 3))
            return cls(data)

    def __init__(self, points):
        self.points = points

    def __len__(self):
        return len(self.points)

    def horizontal_distribution_evenness(self) -> float:
        """Calculate the evenness of the horizontal distribution of points.

        Returns:
            A float between 0 and 1. 1 means that the points are evenly distributed
            horizontally, 0 means that they are not.
        """
        GRID_SIZE = 20
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        bounding_box = np.array([
            [np.min(self.points[:, 0]), np.max(self.points[:, 0])],
            [np.min(self.points[:, 1]), np.max(self.points[:, 1])]
        ])
        bounding_box_size = bounding_box[:, 1] - bounding_box[:, 0]
        cell_size = bounding_box_size / GRID_SIZE
        containing_cell = np.floor((self.points[:, :2] - bounding_box[:, 0]) / cell_size).astype(np.int32)
        # replace points that are exactly on the upper bound of the bounding box
        # with the last cell
        containing_cell[containing_cell == GRID_SIZE] = GRID_SIZE - 1

        for cell_coordinates in containing_cell:
            grid[cell_coordinates[0], cell_coordinates[1]] += 1

        return np.count_nonzero(grid) / (GRID_SIZE * GRID_SIZE)


    def kolmogorov_smirnov_z(self) -> float:
        """Calculate the normality of the z axis of the point cloud using the Kolmogorov-Smirnov test.
        Returns:
            ""A float between 0 and 1. 1 means that the points are normally distributed
        """

        result = stats.kstest(self.points[:, 2], 'norm')
        return result[1]  # [1] refers to the p-value.
    
    def squareness(self) -> float:
        """Calculate the squareness of the bounding box of the point cloud.
        Returns:
            A float between 0 and 1. 1 means that the bounding box is a perfect
            square, 0 means it is a line."""
        x_diff = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
        y_diff = np.max(self.points[:, 1]) - np.min(self.points[:, 1])
        squareness_value = np.min([x_diff,y_diff])/np.max([x_diff,y_diff])
        return squareness_value
    
    def avg_distance(self) -> float:
        """Calculate the average distance from every point of the point cloud
        to every other point.
        Returns:
            A float greater than zero."""
        total_distance = 0
        for i in range(len(self.points)):
            distance = 0
            for j in range(len(self.points)):
                x_diff = self.points[i][0] - self.points[j][0]
                y_diff = self.points[i][1] - self.points[j][1]
                z_diff = self.points[i][2] - self.points[j][2]
                diff = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
                distance += diff
            total_distance += np.average(distance)
        return np.average(total_distance)
    
    def points_per_sqm(self) -> float:
        """Calculate the average number of points per square meter of the
        area of the point cloud's bounding box.
        Returns:
            A float grearter than zero."""
        x_min = np.min(self.points[:, 0])
        x_max = np.max(self.points[:, 0])
        y_min = np.min(self.points[:, 1])
        y_max = np.max(self.points[:, 1])
        quad_points = np.array([[x_min,y_min], [x_max, y_min], [x_min, y_max], [x_max,  y_max]])
        x_up = np.roll(quad_points[:, 0], -1)
        y_down = np.roll(quad_points[:, 1], 1)
        first_sum = np.sum(quad_points[:, 0] * x_up - x_up * quad_points[:, 1])
        second_sum = np.sum(quad_points[:, 1] * y_down - y_down * quad_points[:, 0])
        area = np.abs((first_sum + second_sum) / 2)
        no_of_points = len(self.points)
        return no_of_points/area
    
    def avg_height(self) -> float:
        """Calculate the average height (z values) of the points of the point
        cloud.
        Returns:
            A float between 0 and 1. 1 means that the average height is equal
            to the larges height, 0 means it is equal to the lowest.."""
        z_min = np.min(self.points[:, 2])
        z_max = np.max(self.points[:, 2])
        z_range = z_max - z_min
        z_values = self.points[:, 2]
        z_avg = np.average(z_values)
        normalized_avg = (z_avg - z_min)/z_range
        return normalized_avg
        
    def plot_horizontal_distribution(self):
        plt.scatter(self.points[:, 0], self.points[:, 1])
        plt.show()

    def plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2])
        plt.show()
