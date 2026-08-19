from setuptools import find_packages, setup

setup(
    name="gam",
    version="0.1.0",
    description="Grounded action model with a Molmo2 high-level planner",
    packages=find_packages(include=["gam", "gam.*"])
    + [p for p in find_packages(where="third_party")],
    package_dir={"": ".", "wilddet3d": "third_party/wilddet3d", "sam3": "third_party/sam3",
                 "mdm": "third_party/mdm"},
    package_data={"sam3": ["assets/*"]},
    include_package_data=True,
    python_requires=">=3.10",
)
