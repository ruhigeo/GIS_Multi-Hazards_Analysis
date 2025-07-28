 Multi-Hazard Susceptibility Mapping with Python & AHP

Hi! I'm Ruhi — a Geoinformation Science graduate with a passion for using spatial data to make real-world decisions.  
This project is based on my Master's thesis, where I created a multi-hazard susceptibility map using GIS and the Analytic Hierarchy Process (AHP).

GOAL: The goal is To identify areas most at risk from natural hazards by combining physical, environmental, and social data — all in a reproducible, Python-based workflow.

---

What this Project does

This notebook walks you through:

1- Loading multiple geospatial layers (rainfall, slope, rivers, roads, etc.)

2- Aligning rasters to the same spatial reference

3- Applying AHP (Analytic Hierarchy Process) to calculate weights for each hazard factor

4- Combining the layers into one composite risk map

5- Visualizing the results with clear color gradients (green = safe, red = risky)


Data Used

Each of these is a raster `.tif` file representing a hazard-related factor:

| Factor             | Description                  | Weight Role |
|--------------------|------------------------------|-------------|
| Rainfall           | Annual rainfall levels       | Benefit     |
| Slope (DEM-based)  | Derived from elevation       | Cost        |
| River Proximity    | Distance from rivers         | Benefit     |
| Faultline Proximity| Distance from faults         | Cost        |
| Road Proximity     | Accessibility                | Benefit     |
| Cyclone Paths      | Historical storm tracks      | Cost        |
| Cyclone Shelters   | Availability of shelters     | Benefit     |


AHP Explained

AHP is used to assign weights based on expert judgment.  
Here's the pairwise comparison matrix I created after literature review, domain insights and expert advices:

SAMPLES

| Criteria       | Rainfall | Slope | River | Faultline | Roads | CycloneP | CycloneS |
|----------------|----------|-------|--------|------------|--------|-----------|-----------|
|     Rainfall   | 1        | 3     | 3      | 3          | 2      | 2         | 2         |
|     Slope      | 1/3      | 1     | 2      | 3          | 4      | 2         | 2         |
|   (...)        | ...      | ...   | ...    | ...        | ...    | ...       | ...       |

The Consistency Ratio was 0.059 — within acceptable range, meaning the pairwise judgments are reliable.


Results
 
 Final susceptibility map with values normalized between the 2nd and 98th percentiles
 
 Color-coded gradient: green = low risk, red = high risk
 
 Masked by slope raster to exclude irrelevant areas
 
 Easy-to-read layout and reproducible steps



Tools Used

Python (Jupyter Notebook)

Rasterio for raster processing

NumPy for calculations

Matplotlib for visualization

AHP math logic for weighting



How to Run It Yourself

1. Clone this repo or download the notebook

2. Place your `.tif` files inside a folder named `thesis_data/`

3. Run the notebook step-by-step

4. Final map will be shown at the end!



About Me

My name is Ruhi, and I'm a GIS enthusiast with a background in hazard modeling and data analysis.  
Currently, I work at ROSEN Group and,  I explore the intersection of geospatial intelligence and real-world applications.

This portfolio is my way of sharing how we can turn raw spatial data into meaningful insights — one pixel at a time.



Let’s Connect!

 GitHub:  [@ruhigeo](https://github.com/ruhigeo)
 
 Email:   r.begumnl@gmail.com 
 
 LikedIn: linkedin.com/in/ruhi-begum


 “Maps are not just tools — they’re stories waiting to be told.”


