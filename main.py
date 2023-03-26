from pointcloud import PointCloud

def main():
    cloud = PointCloud.from_file('data/pointclouds/000.xyz')
    print(len(cloud))

if __name__ == '__main__':
    main()