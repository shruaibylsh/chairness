# Chairness: A Web Application for Generating Unique Chair Designs

## Overview

**Chairness** is a web application designed to generate images of unique and creative chair designs. The application leverages a combination of web-scraped chair images from e-commerce platforms and synthetically generated chair images from 3D models using automated Blender scripts. These images are used to train a diffusion network, which powers the image generation process.

### Key Features:
- **Web-Scraped Dataset**: Chair images are collected from various e-commerce platforms to ensure diversity and realism.
- **Synthetic Data Generation**: Automated Blender scripts create additional chair images from 3D models, enhancing the dataset's variety.
- **Diffusion Network**: A state-of-the-art diffusion model is trained on the combined dataset to generate unique chair designs.

### Repository
You can explore the project's code and documentation on GitHub:  
[Chairness GitHub Repository](https://github.com/shruaibylsh/chairness)

---
## How It Works
1. Data Collection:
- Chair images are collected from e-commerce platforms using web scraping.
- Additional synthetic images are generated from 3D models using Blender scripts.

2. Image Post-Processing:
- The background of each image is removed using OpenCV's GrabCut algorithm.
- Processed images are placed on a white background for consistency.

3. Model Training:
- A diffusion model is trained on the processed dataset to learn chair design features.
- Users interact with the web application to generate unique chair designs.