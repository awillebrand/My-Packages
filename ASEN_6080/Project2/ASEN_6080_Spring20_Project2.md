# ASEN 6080: Statistical Orbit Determination
## Project 2 - Interplanetary Navigation

**Due: 1:00pm MST, April 23, 2026 (no exceptions!)**
**Prof. Jay McMahon**

In this project, you will process data to estimate the state and covariance of an Earth flyby. Unlike previous homework assignments, you will not know the true trajectory, and in some cases will not know what errors may be in your filter models or data. The intent of this project is to give you the flavor of a real world orbit determination scenario/process. Most importantly, bragging rights are on the line (as explained in lecture)!

---

## 1. The Flyby Scenario [10 points]

In the following two sections, there are two different data sets to fit. This section explains the common parameters and scenario set-up issues that are common between both problems.

### a. Dynamics

This scenario uses only point mass gravity from the Earth and Sun, and cannonball solar radiation pressure. Note the following for your simulation:

The equations for third body gravity (from the Sun in this case) and solar radiation pressure were in our early lecture on dynamics. Please do double- and triple-check your units when adding these perturbations.

1. Integrate with respect to the center of the Earth (**HINT:** integrate the full reference trajectory in one call of the integrator before processing data with outputs at all observation times in order to avoid numerical issues)
2. Initial Epoch in JD is `2456296.25`
3. Gravitational parameter of the sun is `132712440017.987 km³/s²`
4. Solar pressure flux is `1357 W/m²` at 1 AU (so divide this value by 1 AU in km² to get Φ from the slides)
5. SRP Area to Mass ratio is `0.01 m²/kg`
6. The ephemeris of the Earth around the Sun is described in the document `EphemerisMeeus.pdf` in the Canvas Project 2 Assignment description. The ephemeris that is provided is in EME2000 where the Z-axis is aligned with the Earth's rotation pole. You are welcome to use/pillage the Matlab code `Ephem.m` on Canvas if you want. If you use your own code, note that this data assumed **1 AU = 149597870.7 km**!

### b. Measurements

The measurements for this scenario are range and range-rate measurements taken from the three Deep Space Network (DSN) stations at the locations noted below. The noise on the observations is **5 m on range** and **0.5 mm/s on range-rate**.

The stations move assuming the Earth spins about its z-axis with an initial spin angle of 0° and a rotation rate of `7.29211585275553e-005 rad/sec`. The Earth is assumed to be spherical with radius of `6378.1363 km`.

| Station | Latitude | Longitude | Altitude |
|---|---|---|---|
| Canberra (DSS 34) | -35.398333° | 148.981944° | 0.691750 km |
| Madrid (DSS 65) | 40.427222° | 355.749444° | 0.834539 km |
| Goldstone (DSS 13) | 35.247164° | 243.205° | 1.07114904 km |

> **Note:** For Part 2, the example truth data for the Madrid station is generated using longitude = `-355.749444°`. Use this 'negative value' when comparing to the truth data only. Part 3 uses the 'positive' value specified.

---

## 2. Estimate the State with a Known Target and Models [30 points]

In this data set, the scenario is known and only the state below must be estimated to fit the data. The final target on the B-plane is given so that the filter performance can be judged.

a. Process DSN measurements of the spacecraft, which is inbound to an Earth flyby.

b. The observation data is contained in the file `Project2a_Obs.txt` in the Canvas Project 2 assignment. Note the format on the first line of the file. If no observation is available from a station, the file includes a `NaN` entry.

d. The filter should estimate the position, velocity (with respect to the Earth), and C_R from the cannonball model. The a priori state should be:

| Parameter | Value |
|---|---|
| X | -274096790.0 km |
| Y | -92859240.0 km |
| Z | -40199490.0 km |
| VX | 32.67 km/sec |
| VY | -8.94 km/sec |
| VZ | -3.88 km/sec |
| C_R | 1.2 |

e. The a priori covariance should be set as a diagonal with entries of:

| Parameter | 1-σ Value |
|---|---|
| Position | 100 km |
| Velocity | 0.1 km/sec |
| C_R | 0.1 |

f. Plot the 3-σ covariance envelopes versus time for all estimated states.

g. Plot the pre-fit and post-fit residuals.

h. Plot the 3-σ covariance envelope and the estimate on the B-plane after processing 50, 100, 150, and 200 days of data (so you will have 4 ellipses on the plot). For reference, the true B-plane target (computed at 3 R_SOI, as discussed in extra B-plane handout, `Bplane_intro.pdf`) is:

- **B · R̂** = 14970.824 km
- **B · T̂** = 9796.737 km

Be sure to include discussion of what you see and why it does (or does not) make sense in terms of the accuracy of your final answer.

> **Note:** The true trajectory for the first 50 days, including the STM, is available on Canvas, in order for you to verify your dynamics are correct. The filename is `Project2_Prob2_truth_traj_50days.mat`.

---

## 3. Fit Data with Unknown Issues and no Truth [40 points]

In this data set, you will try to fit a set of observations as precisely as possible using any filtering methodology you prefer. Unlike the previous data set, errors exist in the models that are unknown to you — you must figure them out along the way! No information about the true solution is provided, so you must use your own ingenuity to determine if you are getting a good solution.

a. Process the observation data contained in the file `Project2b_Obs.txt` in the Canvas Project 2 assignment. Note the format on the first line of the file. If no observation is available from a station, the file includes a `NaN` entry.

b. The only sources of gravity in this scenario are from the Earth and Sun as point masses. You will not need to add gravity from any other body, or other gravity coefficients such as J2. There is also solar radiation pressure acting on the spacecraft. Outside of those constraints, anything could be happening (e.g. constants may not be right, etc.) and it is up to you to determine!

d. Your filter should estimate (at a minimum) the position, velocity with respect to the Earth, and C_R from the cannonball model. The a priori state should be:

| Parameter | Value |
|---|---|
| X | -274096770.76544 km |
| Y | -92859266.4499061 km |
| Z | -40199493.6677441 km |
| VX | 32.6704564599943 km/sec |
| VY | -8.93838913761049 km/sec |
| VZ | -3.87881914050316 km/sec |
| C_R | 1.0 |

e. The suggested a priori covariance is a diagonal with entries of:

| Parameter | 1-σ Value |
|---|---|
| Position | 100 km |
| Velocity | 0.1 km/sec |
| C_R | 0.1 |

f. In your write-up, provide the following:

1. An explanation of the filtering methodology used to fit the data.
2. The estimate of the state at the epoch (the initial time).
3. The B-plane target estimate and covariance, in both tabular and plot form.
4. Discuss all relevant details necessary to fully explain the performance of the filter, and how confident you are that the solution is accurate.

---

## 4. Additional Analysis [20 points]

Carry out some additional analysis, which is up to you. A couple ideas:

- Compare results from at least one different filter
- Investigate how the B-plane parameters change if you compute them from a different radius (instead of 3 R_SOI)
- Incorporate consider analysis
- Discuss how you can apply lessons learned to other problems of interest to you
- Re-make the scenario including the moon
- Investigate the effect of an uncertain ephemeris model

Given that this is 20% of your project, there should be some actual thought and analysis put into this section!

---

## 5. Explanation of Deliverables

Everyone will turn in a **written report (PDF)**, a **ZIP file of your code**, and a **solution text (TXT) file** as separate uploads to Canvas by the due date.

The report will document everything done, especially focused on Problems 3 and 4. This will be the ultimate source for the document grade.

The text file mentioned above will be your best estimate of the B-plane for Problem 3 in a plain text file, with a single line containing two numbers: **B · R̂**, a space, then **B · T̂**. The filename should be `<lastname>_<firstname>_bplane.txt`.
