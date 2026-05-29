# Dam Filtration and Pollution Transport Simulator

![Static Badge](https://img.shields.io/badge/python-3.14.3-green)
![Static Badge](https://img.shields.io/badge/numpy-2.4.6-blue)
![Static Badge](https://img.shields.io/badge/scipy-1.17.1-yellow)
![Static Badge](https://img.shields.io/badge/matplotlib-3.10.9-orange)

## About the Project
A desktop application for simulating water seepage and pollution transport under hydraulic structures. The software calculates hydraulic head distribution, groundwater velocity fields, and tracks the spread of contaminants over time using numerical methods.

## Core Technologies
The application is built entirely in Python. The graphical user interface is developed with **Tkinter**. Mathematical modeling, sparse linear algebra, and Delaunay triangulation for adaptive mesh generation rely on **NumPy** and **SciPy**. All dynamic visual representations and interactive timelines are rendered using **Matplotlib**.

## Key Features
The program supports two numerical approaches: the Finite Element Method (FEM) and the Finite Difference Method (FDM). Users can customize the physical domain, dam geometry, and insert multiple sheet piles to analyze different structural behaviors. 

It allows precise configuration of physical properties such as the filtration coefficient, diffusion rate, porosity, and hydraulic head. The simulation tracks continuous or temporary pollution sources asynchronously, allowing users to view the resulting data through interactive, detachable plots with time sliders and mesh toggles.

## Getting Started
Download the standalone executable from the **Releases** section to run the application immediately without any installation. 

To run from the source code:
1. Clone the repository.
2. Install the required dependencies: `pip install numpy scipy matplotlib`
3. Launch the application: `python Diploma.py`
