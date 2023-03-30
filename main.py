import numpy as np

from pointcloud import PointCloud
import matplotlib.pyplot as plt


def main():
    labels = ["building", "car", "fence", "pole", "tree"]
    horizontal_distribution_data = {label: [] for label in labels}
    normality_xy_data = {label: [] for label in labels}
    normality_z_data = {label: [] for label in labels}
    variance_xy_data = {label: [] for label in labels}
    variance_z_data = {label: [] for label in labels}
    kurtosis_xy_data = {label: [] for label in labels}
    kurtosis_z_data = {label: [] for label in labels}
    skewness_xy_data = {label: [] for label in labels}
    skewness_z_data = {label: [] for label in labels}
    squareness_xy_data = {label: [] for label in labels}
    curviness_data = {label: [] for label in labels}
    amount_of_points_data = {label: [] for label in labels}
    average_distance_data = {label: [] for label in labels}
    points_per_m2_data = {label: [] for label in labels}
    average_height_data = {label: [] for label in labels}

    for hundred in range(5):
        print(f"Processing {labels[hundred]}")
        horizontal_data = horizontal_distribution_data[labels[hundred]]
        normality_xy = normality_xy_data[labels[hundred]]
        normality_z = normality_z_data[labels[hundred]]
        variance_xy = variance_xy_data[labels[hundred]]
        variance_z = variance_z_data[labels[hundred]]
        kurtosis_xy = kurtosis_xy_data[labels[hundred]]
        kurtosis_z = kurtosis_z_data[labels[hundred]]
        skewness_xy = skewness_xy_data[labels[hundred]]
        skewness_z = skewness_z_data[labels[hundred]]
        squareness_xy = squareness_xy_data[labels[hundred]]
        curviness = curviness_data[labels[hundred]]
        amount_of_points = amount_of_points_data[labels[hundred]]
        average_distance = average_distance_data[labels[hundred]]
        points_per_m2 = points_per_m2_data[labels[hundred]]
        average_height = average_height_data[labels[hundred]]
        for number in range(100):
            cloud = PointCloud.from_file(f"data/pointclouds/{hundred}{number:02d}.xyz")
            evenness = cloud.horizontal_distribution_evenness()
            horizontal_data.append(evenness)
            normality_xy.append(cloud.kolmogorov_smirnov_xy())
            normality_z.append(cloud.kolmogorov_smirnov_z())
            variance_xy.append(cloud.variance_xy())
            variance_z.append(cloud.variance_z())
            kurtosis_xy.append(cloud.kurtosis_xy())
            kurtosis_z.append(cloud.kurtosis_z())
            skewness_xy.append(cloud.skew_xy())
            skewness_z.append(cloud.skew_z())
            squareness_xy.append(cloud.squareness())
            curviness.append(cloud.curviness())
            amount_of_points.append(cloud.amount_of_points())
            average_distance.append(cloud.avg_distance())
            points_per_m2.append(cloud.points_per_sqm())
            average_height.append(cloud.avg_height())


        print(f"Average {hundred}: {sum(horizontal_data) / len(horizontal_data)}")
        print(f"Standard deviation: {hundred}: {np.std(horizontal_data)}")
        print(f"Average normality: {hundred}: {sum(normality_xy) / len(normality_xy)}")
        print(f"Standard deviation: {hundred}: {np.std(normality_xy)}")

        horizontal_distribution_data[labels[hundred]] = horizontal_data
        normality_xy_data[labels[hundred]] = normality_xy
        normality_z_data[labels[hundred]] = normality_z
        variance_xy_data[labels[hundred]] = variance_xy
        variance_z_data[labels[hundred]] = variance_z
        kurtosis_xy_data[labels[hundred]] = kurtosis_xy
        kurtosis_z_data[labels[hundred]] = kurtosis_z
        skewness_xy_data[labels[hundred]] = skewness_xy
        skewness_z_data[labels[hundred]] = skewness_z
        squareness_xy_data[labels[hundred]] = squareness_xy
        curviness_data[labels[hundred]] = curviness
        amount_of_points_data[labels[hundred]] = amount_of_points
        # average_distance_data[labels[hundred]] = average_distance
        points_per_m2_data[labels[hundred]] = points_per_m2
        average_height_data[labels[hundred]] = average_height


    make_boxplot(
        horizontal_distribution_data, "Horizontal Distribution Evenness by Class"
    )
    make_boxplot(normality_xy_data, "Normality XY by Class")
    make_boxplot(normality_z_data, "Normality Z by Class")
    make_boxplot(variance_xy_data, "Variance XY by Class")
    make_boxplot(variance_z_data, "Variance Z by Class")
    make_boxplot(kurtosis_xy_data, "Kurtosis XY by Class")
    make_boxplot(kurtosis_z_data, "Kurtosis Z by Class")
    make_boxplot(skewness_xy_data, "Skewness XY by Class")
    make_boxplot(skewness_z_data, "Skewness Z by Class")
    make_boxplot(squareness_xy_data, "Squareness XY by Class")
    make_boxplot(curviness_data, "Curviness by Class")
    make_boxplot(amount_of_points_data, "Amount of points by Class")
    make_boxplot(average_distance_data, "Average distance by Class")
    make_boxplot(points_per_m2_data, "Points per m2 by Class")
    make_boxplot(average_height_data, "Average height by Class")


def make_boxplot(all_data: dict[str, list[float]], title: str):
    plt.boxplot(all_data.values())
    plt.xticks(range(1, len(all_data) + 1), all_data.keys())
    plt.xlabel("Class")
    plt.ylabel("Return value")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()
