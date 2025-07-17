import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Core RK4 Solver ---

def runge_kutta_4(f, y0, t0, tf, h, args=()):
    """
    Fourth-order Runge-Kutta method for solving systems of ODEs.

    Args:
        f (callable): The function that defines the system of ODEs.
        y0 (list or np.ndarray): The initial conditions.
        t0 (float): The initial time.
        tf (float): The final time.
        h (float): The step size.
        args (tuple, optional): Additional arguments to pass to the function f.

    Returns:
        tuple: A tuple containing the time points (np.ndarray) and the
               solution (np.ndarray).
    """
    t = np.arange(t0, tf, h)
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(n - 1):
        k1 = h * f(y[i], t[i], *args)
        k2 = h * f(y[i] + 0.5 * k1, t[i] + 0.5 * h, *args)
        k3 = h * f(y[i] + 0.5 * k2, t[i] + 0.5 * h, *args)
        k4 = h * f(y[i] + k3, t[i] + h, *args)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t, y

# --- ODE System Definitions ---

def lotka_volterra(state, t, alpha, beta, delta, gamma):
    """Lotka-Volterra predator-prey model."""
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return np.array([dxdt, dydt])

def lorenz_attractor(state, t, sigma, rho, beta):
    """Lorenz attractor model, a classic chaotic system."""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def van_der_pol(state, t, mu):
    """Van der Pol oscillator, a non-conservative oscillator."""
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return np.array([dxdt, dvdt])

def damped_harmonic_oscillator(state, t, damping, omega):
    """Damped harmonic oscillator model."""
    x, v = state
    dxdt = v
    dvdt = -damping * v - omega**2 * x
    return np.array([dxdt, dvdt])

def particle_motion(state, t, g, k):
    """Projectile motion with air resistance."""
    x, y, vx, vy = state
    v_total = np.sqrt(vx**2 + vy**2)
    dxdt = vx
    dydt = vy
    dvxdt = -k * v_total * vx
    dvydt = -g - k * v_total * vy
    return np.array([dxdt, dydt, dvxdt, dvydt])

def rocket_launch(state, t, M_earth, R_earth, thrust, burn_rate, initial_mass):
    """Simplified 1D rocket launch to the moon."""
    h, v = state
    G = 6.67430e-11  # Gravitational constant
    
    # Current mass of the rocket
    current_mass = initial_mass - burn_rate * t
    if current_mass <= (initial_mass * 0.2): # Assume 80% is fuel
        thrust = 0
        current_mass = initial_mass * 0.2

    # Gravitational acceleration at height h
    g_h = (G * M_earth) / (R_earth + h)**2
    
    # Acceleration
    acceleration = -g_h
    if thrust > 0 and current_mass > 0:
        acceleration += thrust / current_mass

    dhdt = v
    dvdt = acceleration
    return np.array([dhdt, dvdt])


# --- Streamlit App ---

st.set_page_config(layout="wide")

st.title("🚀 Runge-Kutta 4th Order Simulation Hub")
st.markdown("""
This interactive dashboard uses the **Runge-Kutta 4th order method** to simulate and visualize various systems of ordinary differential equations.
Select a model from the sidebar and adjust its parameters to see how the system behaves.
""")

# --- Sidebar for Model Selection and Parameters ---
st.sidebar.header("Simulation Setup")

model_choice = st.sidebar.selectbox(
    "Select a model to simulate:",
    ("Particle Shot", "Rocket Launch", "Lotka-Volterra (Predator-Prey)", "Lorenz Attractor", "Van der Pol Oscillator", "Damped Harmonic Oscillator")
)

# --- Simulation and Visualization Logic ---

if model_choice == "Particle Shot":
    st.header("🔫 Particle Shot Simulation")
    st.markdown("This model simulates the trajectory of a projectile (like a photon or particle) fired with an initial velocity and angle, considering gravity and simple air resistance.")

    # Parameters
    st.sidebar.markdown("### Model Parameters")
    g = st.sidebar.slider("Gravity (g)", 1.0, 20.0, 9.8, 0.1)
    k_drag = st.sidebar.slider("Air Drag Coefficient (k)", 0.0, 0.1, 0.01, 0.001)

    # Initial Conditions
    st.sidebar.markdown("### Initial Conditions")
    v0 = st.sidebar.slider("Initial Velocity (v₀)", 10.0, 200.0, 100.0, 1.0)
    angle_deg = st.sidebar.slider("Launch Angle (degrees)", 0, 90, 45, 1)
    angle_rad = np.deg2rad(angle_deg)
    
    y0 = [0, 0, v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)] # x, y, vx, vy

    # Simulation Settings
    st.sidebar.markdown("### Simulation Settings")
    t_final = st.sidebar.slider("Simulation Time (t)", 5, 100, 20, 1)
    h_step = st.sidebar.slider("Step Size (h)", 0.001, 0.1, 0.01, 0.001)

    # Run Simulation
    t_points, solution = runge_kutta_4(particle_motion, y0, 0, t_final, h_step, args=(g, k_drag))
    
    # Filter data until particle hits the ground
    ground_hit_index = np.where(solution[:, 1] < 0)[0]
    if len(ground_hit_index) > 0:
        solution = solution[:ground_hit_index[0]]
        t_points = t_points[:ground_hit_index[0]]

    # Plotting
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(x=solution[:, 0], y=solution[:, 1], mode='lines', name='Trajectory', line=dict(color='orange')))
    fig_traj.update_layout(title="Particle Trajectory", xaxis_title="Horizontal Distance (m)", yaxis_title="Altitude (m)", yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig_traj, use_container_width=True)

    df = pd.DataFrame({'Time': t_points, 'x': solution[:, 0], 'y': solution[:, 1], 'vx': solution[:, 2], 'vy': solution[:, 3]})


elif model_choice == "Rocket Launch":
    st.header("🚀 Rocket Launch Simulation (1D)")
    st.markdown("This model simulates a vertical rocket launch from the Earth's surface. It accounts for the change in gravity with altitude and the decrease in rocket mass as fuel is burned.")

    # Constants
    M_earth = 5.972e24 # kg
    R_earth = 6.371e6  # m

    # Parameters
    st.sidebar.markdown("### Rocket Parameters")
    initial_mass = st.sidebar.slider("Initial Rocket Mass (kg)", 100000, 3000000, 500000, 10000)
    thrust = st.sidebar.slider("Thrust (N)", 1e6, 35e6, 7.5e6, 1e5)
    burn_rate = st.sidebar.slider("Fuel Burn Rate (kg/s)", 500, 5000, 2000, 100)

    # Initial Conditions
    st.sidebar.markdown("### Initial Conditions")
    y0 = [0, 0] # h, v

    # Simulation Settings
    st.sidebar.markdown("### Simulation Settings")
    t_final = st.sidebar.slider("Simulation Time (t)", 100, 2000, 500, 10)
    h_step = st.sidebar.slider("Step Size (h)", 0.01, 1.0, 0.1, 0.01)

    # Run Simulation
    t_points, solution = runge_kutta_4(rocket_launch, y0, 0, t_final, h_step, args=(M_earth, R_earth, thrust, burn_rate, initial_mass))

    # Plotting
    fig_alt = go.Figure()
    fig_alt.add_trace(go.Scatter(x=t_points, y=solution[:, 0] / 1000, mode='lines', name='Altitude', line=dict(color='cyan')))
    fig_alt.update_layout(title="Rocket Altitude vs. Time", xaxis_title="Time (s)", yaxis_title="Altitude (km)")
    st.plotly_chart(fig_alt, use_container_width=True)
    
    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(x=t_points, y=solution[:, 1] / 1000, mode='lines', name='Velocity', line=dict(color='lime')))
    fig_vel.update_layout(title="Rocket Velocity vs. Time", xaxis_title="Time (s)", yaxis_title="Velocity (km/s)")
    st.plotly_chart(fig_vel, use_container_width=True)

    df = pd.DataFrame({'Time (s)': t_points, 'Altitude (m)': solution[:, 0], 'Velocity (m/s)': solution[:, 1]})


elif model_choice == "Lotka-Volterra (Predator-Prey)":
    st.header("Lotka-Volterra (Predator-Prey) Model")
    st.markdown("This model describes the dynamics of biological systems in which two species interact, one as a predator and the other as prey.")
    
    # Parameters
    st.sidebar.markdown("### Model Parameters")
    alpha = st.sidebar.slider("α (Prey Growth Rate)", 0.1, 2.0, 1.1, 0.01)
    beta = st.sidebar.slider("β (Predation Rate)", 0.1, 2.0, 0.4, 0.01)
    delta = st.sidebar.slider("δ (Predator Growth Rate)", 0.1, 2.0, 0.4, 0.01)
    gamma = st.sidebar.slider("γ (Predator Death Rate)", 0.1, 2.0, 1.1, 0.01)
    
    # Initial Conditions
    st.sidebar.markdown("### Initial Conditions")
    y0 = [st.sidebar.number_input("Initial Prey Population", value=10.0),
          st.sidebar.number_input("Initial Predator Population", value=10.0)]
    
    # Simulation Settings
    st.sidebar.markdown("### Simulation Settings")
    t_final = st.sidebar.slider("Simulation Time (t)", 10, 500, 100, 10)
    h_step = st.sidebar.slider("Step Size (h)", 0.001, 0.1, 0.01, 0.001)

    # Run Simulation
    t_points, solution = runge_kutta_4(lotka_volterra, y0, 0, t_final, h_step, args=(alpha, beta, delta, gamma))
    
    # Plotting
    fig_pop = go.Figure()
    fig_pop.add_trace(go.Scatter(x=t_points, y=solution[:, 0], mode='lines', name='Prey', line=dict(color='royalblue')))
    fig_pop.add_trace(go.Scatter(x=t_points, y=solution[:, 1], mode='lines', name='Predators', line=dict(color='crimson')))
    fig_pop.update_layout(title="Population Dynamics Over Time", xaxis_title="Time", yaxis_title="Population")
    st.plotly_chart(fig_pop, use_container_width=True)

    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=solution[:, 0], y=solution[:, 1], mode='lines', name='Phase Portrait', line=dict(color='purple')))
    fig_phase.update_layout(title="Phase Portrait (Predator vs. Prey)", xaxis_title="Prey Population", yaxis_title="Predator Population")
    st.plotly_chart(fig_phase, use_container_width=True)
    
    df = pd.DataFrame({'Time': t_points, 'Prey': solution[:, 0], 'Predators': solution[:, 1]})


elif model_choice == "Lorenz Attractor":
    st.header("Lorenz Attractor")
    st.markdown("A classic example of a system that exhibits chaotic behavior. The trajectory forms a distinctive butterfly shape but never repeats itself.")
    
    # Parameters
    st.sidebar.markdown("### Model Parameters")
    sigma = st.sidebar.slider("σ (Sigma)", 1.0, 20.0, 10.0, 0.1)
    rho = st.sidebar.slider("ρ (Rho)", 1.0, 50.0, 28.0, 0.1)
    beta = st.sidebar.slider("β (Beta)", 0.1, 5.0, 8.0/3.0, 0.01)
    
    # Initial Conditions
    st.sidebar.markdown("### Initial Conditions")
    y0 = [st.sidebar.number_input("Initial X", value=0.0),
          st.sidebar.number_input("Initial Y", value=1.0),
          st.sidebar.number_input("Initial Z", value=1.05)]
          
    # Simulation Settings
    st.sidebar.markdown("### Simulation Settings")
    t_final = st.sidebar.slider("Simulation Time (t)", 10, 100, 50, 5)
    h_step = st.sidebar.slider("Step Size (h)", 0.001, 0.1, 0.01, 0.001)

    # Run Simulation
    t_points, solution = runge_kutta_4(lorenz_attractor, y0, 0, t_final, h_step, args=(sigma, rho, beta))

    # Plotting
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=solution[:, 0], y=solution[:, 1], z=solution[:, 2],
        mode='lines',
        line=dict(color='blue', width=2)
    )])
    fig_3d.update_layout(title="Lorenz Attractor Phase Portrait", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    st.plotly_chart(fig_3d, use_container_width=True)

    df = pd.DataFrame({'Time': t_points, 'x': solution[:, 0], 'y': solution[:, 1], 'z': solution[:, 2]})


elif model_choice == "Van der Pol Oscillator":
    st.header("Van der Pol Oscillator")
    st.markdown("A non-conservative oscillator with non-linear damping. It exhibits a 'limit cycle', a trajectory that the system is attracted to, regardless of the starting point (within a basin of attraction).")

    # Parameters
    st.sidebar.markdown("### Model Parameters")
    mu = st.sidebar.slider("μ (Damping Coefficient)", 0.0, 10.0, 1.0, 0.1)

    # Initial Conditions
    st.sidebar.markdown("### Initial Conditions")
    y0 = [st.sidebar.number_input("Initial Position (x)", value=2.0),
          st.sidebar.number_input("Initial Velocity (v)", value=0.0)]

    # Simulation Settings
    st.sidebar.markdown("### Simulation Settings")
    t_final = st.sidebar.slider("Simulation Time (t)", 10, 200, 50, 5)
    h_step = st.sidebar.slider("Step Size (h)", 0.001, 0.1, 0.01, 0.001)

    # Run Simulation
    t_points, solution = runge_kutta_4(van_der_pol, y0, 0, t_final, h_step, args=(mu,))

    # Plotting
    fig_pop = go.Figure()
    fig_pop.add_trace(go.Scatter(x=t_points, y=solution[:, 0], mode='lines', name='Position (x)', line=dict(color='green')))
    fig_pop.add_trace(go.Scatter(x=t_points, y=solution[:, 1], mode='lines', name='Velocity (v)', line=dict(color='orange')))
    fig_pop.update_layout(title="State Variables Over Time", xaxis_title="Time", yaxis_title="Value")
    st.plotly_chart(fig_pop, use_container_width=True)

    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=solution[:, 0], y=solution[:, 1], mode='lines', name='Phase Portrait', line=dict(color='red')))
    fig_phase.update_layout(title="Phase Portrait (Velocity vs. Position)", xaxis_title="Position (x)", yaxis_title="Velocity (v)")
    st.plotly_chart(fig_phase, use_container_width=True)

    df = pd.DataFrame({'Time': t_points, 'Position (x)': solution[:, 0], 'Velocity (v)': solution[:, 1]})


elif model_choice == "Damped Harmonic Oscillator":
    st.header("Damped Harmonic Oscillator")
    st.markdown("Models a classic physical system like a mass on a spring with friction. Depending on the damping, it can be underdamped (oscillates), critically damped, or overdamped (decays without oscillation).")
    
    # Parameters
    st.sidebar.markdown("### Model Parameters")
    damping = st.sidebar.slider("ζ (Damping Ratio)", 0.0, 5.0, 0.5, 0.05)
    omega = st.sidebar.slider("ω (Natural Frequency)", 0.1, 10.0, 2.0, 0.1)

    # Initial Conditions
    st.sidebar.markdown("### Initial Conditions")
    y0 = [st.sidebar.number_input("Initial Position (x)", value=1.0),
          st.sidebar.number_input("Initial Velocity (v)", value=0.0)]

    # Simulation Settings
    st.sidebar.markdown("### Simulation Settings")
    t_final = st.sidebar.slider("Simulation Time (t)", 10, 100, 20, 1)
    h_step = st.sidebar.slider("Step Size (h)", 0.001, 0.1, 0.01, 0.001)

    # Run Simulation
    t_points, solution = runge_kutta_4(damped_harmonic_oscillator, y0, 0, t_final, h_step, args=(damping, omega))

    # Plotting
    fig_pop = go.Figure()
    fig_pop.add_trace(go.Scatter(x=t_points, y=solution[:, 0], mode='lines', name='Position (x)', line=dict(color='purple')))
    fig_pop.add_trace(go.Scatter(x=t_points, y=solution[:, 1], mode='lines', name='Velocity (v)', line=dict(color='teal')))
    fig_pop.update_layout(title="State Variables Over Time", xaxis_title="Time", yaxis_title="Value")
    st.plotly_chart(fig_pop, use_container_width=True)

    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=solution[:, 0], y=solution[:, 1], mode='lines', name='Phase Portrait', line=dict(color='magenta')))
    fig_phase.update_layout(title="Phase Portrait (Velocity vs. Position)", xaxis_title="Position (x)", yaxis_title="Velocity (v)")
    st.plotly_chart(fig_phase, use_container_width=True)

    df = pd.DataFrame({'Time': t_points, 'Position (x)': solution[:, 0], 'Velocity (v)': solution[:, 1]})

# --- Display Raw Data ---
st.header("Raw Data")
st.markdown("Here is a sample of the generated data for the selected model:")
st.dataframe(df.head())
