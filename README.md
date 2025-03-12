# Aimlytics
Aimlytics is a computer vision-based scoring system designed to analyze target images and compute
precise decimal scores automatically, according to the **ISSF** rules for **C10 and P10 shooting**.

This project was developed with **Opencv** for the _Signal, Image and Video_ exam in the AIS master degree at the University of Trento.


## Features
* Automatic target detection

* **Perspective correction**: dynamically adjusts the target perspective for accurate analysis.

* Bullet hole detection: recognizes and isolates bullet holes using image processing techniques.

* **Scoring computation**


## Configure the project

- Create the virtualenv:
`python3 -m venv .venv`

- Activate it:
`source .venv/bin/activate`

- Install the requisites: `pip install -r requirements.txt`

- Set the interpreter

## Run the project
There are **2 ways** to run Aimlytics:
- Use the **Jupyter notebook** file changing the input image to analyze
- Run the **Python main**. You can stop the running program by pressing any keyboard key

## Authors

**Alessandra Benassi** and **Irene Verria** - DISI, University of Trento
