import numpy as np
from matplotlib import pyplot as plt


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


    def plot_horizontal_distribution(self):
        plt.scatter(self.points[:, 0], self.points[:, 1])
        plt.show()

    def plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2])
        plt.show()