import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy._typing import NDArray
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

TIMING_ENABLED = False

# decorator for timing
def timeit(method):
    if not TIMING_ENABLED:
        return method

    def timed(*args, **kw):
        import time

        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print(f"{method.__name__} took {te - ts:02f} seconds")
        return result

    return timed

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

    def _compute_normals(self, k=10):
        # Find the k nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(self.points)
        distances, indices = nbrs.kneighbors(self.points)

        normals = []

        for idx in indices:
            # Calculate the covariance matrix of the neighbors
            neighbors = self.points[idx]
            cov_matrix = np.cov(neighbors.T)

            # Extract the eigenvector corresponding to the smallest eigenvalue as the normal
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            normal = eigenvectors[:, np.argmin(eigenvalues)]

            normals.append(normal)

        return np.array(normals)

    @timeit
    def curviness(self, k=10):
        normals = self._compute_normals(k)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(normals)
        distances, indices = nbrs.kneighbors(normals)

        angular_diffs = []

        for i, idx in enumerate(indices):
            normal1 = normals[i]
            neighbor_normals = normals[idx]
            for normal2 in neighbor_normals:
                # Compute the dot product between the two normals
                dot_product = np.dot(normal1, normal2)

                # Clamp the dot product to the range [-1, 1] to avoid numerical issues
                dot_product = np.clip(dot_product, -1, 1)

                # Compute the angular difference in radians and convert it to degrees
                angle = np.arccos(dot_product)
                angle_degrees = np.degrees(angle)

                angular_diffs.append(angle_degrees)

        # Calculate the curviness as the average of the angular differences
        curviness = np.mean(angular_diffs)

        # Normalize curviness to the range [0, 1]
        curviness_normalized = (curviness - 0) / 180

        return curviness_normalized
    @timeit
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

    @timeit
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
    @timeit
        """Calculate the amount of points in the point cloud.
        Returns:
            An integer representing the amount of points in the point cloud.
        """
        return len(self.points)

    @timeit
    def plot_horizontal_distribution(self):
        plt.scatter(self.points[:, 0], self.points[:, 1])
        plt.show()


    @timeit
    def variance_xy(self) -> float:
        """Calculate the variance of the horizontal distribution of points.
        Returns:
            A float representing the variance of the horizontal distribution of points.
        """
        # fixme returns non scalar
        return 0.0
        #return float(np.var(self.points[:, :2], axis=0))

    @timeit
    def variance_z(self) -> float:
        """Calculate the variance of the vertical distribution of points.
        Returns:
            A float representing the variance of the vertical distribution of points.
        """
        return np.var(self.points[:, 2], axis=0)


    @timeit
    def kurtosis_xy(self) -> float:
        """Calculate the kurtosis of the horizontal distribution of points.
        Returns:
            A float representing the kurtosis of the horizontal distribution of points.
        """
        # fixme returns non-scalar
        return 0.0
        #return scipy.stats.kurtosis(self.points[:, :2], axis=0, fisher=False, bias=True)

    @timeit
    def kurtosis_z(self) -> float:
        """Calculate the kurtosis of the vertical distribution of points.
        Returns:
            A float representing the kurtosis of the vertical distribution of points.
        """
        return scipy.stats.kurtosis(self.points[:, 2], axis=0, fisher=False, bias=True)

    @timeit
    def skew_xy(self) -> float:
        """Calculate the skewness of the horizontal distribution of points.
        Returns:
            A float representing the skewness of the horizontal distribution of points.
        """
        # fixme returns non-scalar
        return 0.0
        #return scipy.stats.skew(self.points[:, :2], axis=0, bias=True)

    @timeit
    def skew_z(self) -> float:
        """Calculate the skewness of the vertical distribution of points.
        Returns:
            A float representing the skewness of the vertical distribution of points.
        """
        return scipy.stats.skew(self.points[:, 2], axis=0, bias=True)
    
    @timeit
    def squareness(self) -> float:
        """Calculate the squareness of the bounding box of the point cloud.
        Returns:
            A float between 0 and 1. 1 means that the bounding box is a perfect
            square, 0 means it is a line."""
        x_diff = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
        y_diff = np.max(self.points[:, 1]) - np.min(self.points[:, 1])
        squareness_value = np.min([x_diff,y_diff])/np.max([x_diff,y_diff])
        return squareness_value
    
    @timeit
    def avg_distance(self) -> float:
        """Calculate the average distance from every point of the point cloud
        to every other point.
        Returns:
            A float greater than zero."""
        # fixme, takes too long. 13 seconds per point cloud. All other features take less than 1 second. It's O(n^2)
        return 0.0
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
    
    @timeit
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
    
    @timeit
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

    def compute_feature_matrix(self) -> NDArray:
        # Gets all methods of the class that are not private, from, to, plot or compute_feature_matrix
        methods = [
            method
            for method in dir(self)
            if callable(getattr(self, method))
            and not method.startswith(("_", "from", "to", "plot", "compute_feature_matrix"))
        ]

        results = [getattr(self, method)() for method in methods]
        return np.array(results)
