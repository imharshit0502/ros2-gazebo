from glob import glob

from setuptools import find_packages, setup

package_name = "my_robot_perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*")),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/models", glob("models/*")),
        ("share/" + package_name + "/test_images", glob("test_images/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="harshit-jain",
    description="Shelf-void detection for the retail store robot",
    license="TODO",
    entry_points={
        "console_scripts": [
            "void_detector = my_robot_perception.void_detector_node:main",
            "detect_voids_offline = my_robot_perception.detect_voids_offline:main",
        ],
    },
)
