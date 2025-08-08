# Runge-Kutta 4th Order Simulation Hub 🚀

This repository provides an interactive simulation tool built using **Streamlit** that uses the **Runge-Kutta 4th order** method to solve and visualize the solutions of various ordinary differential equations (ODEs). It covers different types of dynamic systems, such as predator-prey models, chaotic attractors, damped oscillators, and rocket launches.

## Streamlit App

You can run the app directly on Streamlit by following this link:  
[Runge-Kutta 4th Order Simulation Hub](https://runge-kutta-order-4-xy6kmxwfmkhvum5wpwwjtr.streamlit.app/)

## Overview

The **Runge-Kutta 4th Order** method is a numerical technique for solving ODEs. This project allows you to select from different models, adjust parameters, and see the simulations and visualizations.

### Available Models:
- **Particle Shot Simulation**: Simulate the trajectory of a particle under gravity and air resistance.
- **Rocket Launch Simulation (1D)**: Simulate a vertical rocket launch considering gravitational acceleration and changing rocket mass.
- **Lotka-Volterra (Predator-Prey) Model**: Simulate the dynamics of predator and prey populations.
- **Lorenz Attractor**: Visualize the chaotic behavior of a three-dimensional system.
- **Van der Pol Oscillator**: Simulate a non-conservative oscillator with nonlinear damping.
- **Damped Harmonic Oscillator**: Simulate a harmonic oscillator with damping.

### Key Features:
- Interactive dashboard with sliders for adjusting parameters.
- Real-time simulation and visualization of dynamic systems.
- Plotting of solutions, including phase portraits, time-series, and 3D visualizations.
- Downloadable raw data for further analysis.

## Installation

To run this app locally, you'll need to have Python installed. You can set up the environment and install the necessary dependencies using `pip`:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/rk4-simulation.git
   cd Runge-Kutta-Order-4
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app::
   ```bash
   streamlit run rk4.py
   ```

## Usage

Once the app is launched, you can:

1. Select the model you wish to simulate from the sidebar.
2. Adjust the parameters using the provided sliders.
3. Watch the simulation run in real-time, and view the results in interactive plots.
4. Download the generated data for further analysis or exploration.

### Sample Outputs

- **Particle Shot Simulation**: Displays a plot showing the trajectory with horizontal distance vs. altitude.
- **Rocket Launch Simulation**: Displays altitude and velocity of the rocket over time.
- **Lotka-Volterra Model**: Shows the population dynamics of predators and prey over time.
- **Lorenz Attractor**: Generates a 3D chaotic phase portrait representing the Lorenz system.
- **Van der Pol Oscillator**: Plots position and velocity over time along with phase portraits.
- **Damped Harmonic Oscillator**: Displays position, velocity, and phase portraits of the system's motion.

## Files

- **rk4.py**: The main Python script that implements the Runge-Kutta 4th Order method and handles the Streamlit app logic.
- **requirements.txt**: A file containing the list of Python dependencies required to run the project.

## Contributing

Feel free to fork this repository and submit pull requests for adding new models, features, or improvements. If you have any suggestions or encounter issues, please open an issue, and I will be happy to assist you!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
