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


    def kolmogorov_smirnov_xy(self) -> float:
        """Calculate the normality of the z axis of the point cloud using the Kolmogorov-Smirnov test.
        Returns:
            ""A float between 0 and 1. 1 means that the points are normally distributed
        """
        #firstly calculate the mean of the point cloud
        xmean = np.mean(self.points[:, 0])
        ymean = np.mean(self.points[:, 1])
        #then calculate the distance of each point to this mean
        distances = np.sqrt((self.points[:, 0] - xmean)**2 + (self.points[:, 1] - ymean)**2)
        result = stats.kstest(distances, 'norm')
        return result[0]  # [0] refers to the test statistic, [1] refers to the p-value.


    def amount_of_points(self) -> int:
        """Calculate the amount of points in the point cloud.
        Returns:
            An integer representing the amount of points in the point cloud.
        """
        return len(self.points)
    def plot_horizontal_distribution(self):
        plt.scatter(self.points[:, 0], self.points[:, 1])
        plt.show()


    def variance_xy(self) -> float:
        """Calculate the variance of the horizontal distribution of points.
        Returns:
            A float representing the variance of the horizontal distribution of points.
        """
        return np.var(self.points[:, :2], axis=0)

    def variance_z(self) -> float:
        """Calculate the variance of the vertical distribution of points.
        Returns:
            A float representing the variance of the vertical distribution of points.
        """
        return np.var(self.points[:, 2], axis=0)


    def kurtosis_xy(self) -> float:
        """Calculate the kurtosis of the horizontal distribution of points.
        Returns:
            A float representing the kurtosis of the horizontal distribution of points.
        """
        return scipy.stats.kurtosis(self.points[:, :2], axis=0, fisher=False, bias=True)

    def kurtosis_z(self) -> float:
        """Calculate the kurtosis of the vertical distribution of points.
        Returns:
            A float representing the kurtosis of the vertical distribution of points.
        """
        return scipy.stats.kurtosis(self.points[:, 2], axis=0, fisher=False, bias=True)

    def skew_xy(self) -> float:
        """Calculate the skewness of the horizontal distribution of points.
        Returns:
            A float representing the skewness of the horizontal distribution of points.
        """
        return scipy.stats.skew(self.points[:, :2], axis=0, bias=True)

    def skew_z(self) -> float:
        """Calculate the skewness of the vertical distribution of points.
        Returns:
            A float representing the skewness of the vertical distribution of points.
        """
        return scipy.stats.skew(self.points[:, 2], axis=0, bias=True)

    def plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2])
        plt.show()
