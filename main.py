import numpy as np

from pointcloud import PointCloud
import pandas as pd
import matplotlib.pyplot as plt


def main():
    labels = ["building", "car", "fence", "pole", "tree"]
    horizontal_distribution_data = {label: [] for label in labels}
    for hundred in range(5):
        print(f"Processing {labels[hundred]}")
        data = horizontal_distribution_data[labels[hundred]]
        for number in range(100):
            cloud = PointCloud.from_file(f"data/pointclouds/{hundred}{number:02d}.xyz")
            evenness = cloud.horizontal_distribution_evenness()
            data.append(evenness)

        print(f"Average {hundred}: {sum(data) / len(data)}")
        print(f"Standard deviation: {hundred}: {np.std(data)}")
        horizontal_distribution_data[labels[hundred]] = data

    make_boxplot(
        horizontal_distribution_data, "Horizontal Distribution Evenness by Class"
    )


def make_boxplot(all_data: dict[str, list[float]], title: str):
    plt.boxplot(all_data.values())
    plt.xticks(range(1, len(all_data) + 1), all_data.keys())
    plt.xlabel("Class")
    plt.ylabel("Return value")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()
