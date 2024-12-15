```python
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class TwoBodySystem:
    def __init__(self, G=6.67430e-11):
        self.G = G
        
    def calculate_orbital_parameters(self, m1, m2, r):
        """Calculate basic orbital parameters"""
        # Force
        force = self.G * m1 * m2 / (r**2)
        
        # Period
        period = 2 * np.pi * np.sqrt(r**3 / (self.G * (m1 + m2)))
        
        # Velocity
        velocity = np.sqrt(self.G * (m1 + m2) / r)
        
        return force, period, velocity
    
    def calculate_positions(self, m1, m2, r, t):
        """Calculate positions of both bodies over time"""
        # Center of mass calculations
        total_mass = m1 + m2
        mu = m2/total_mass
        r1 = r * mu          # Distance of m1 from COM
        r2 = r * (1 - mu)    # Distance of m2 from COM
        
        # Angular velocity
        omega = np.sqrt(self.G * total_mass / r**3)
        
        # Positions
        body1_x = r1 * np.cos(omega * t)
        body1_y = r1 * np.sin(omega * t)
        body2_x = -r2 * np.cos(omega * t)
        body2_y = -r2 * np.sin(omega * t)
        
        return body1_x, body1_y, body2_x, body2_y
    
    def calculate_energies(self, m1, m2, r, v):
        """Calculate kinetic and potential energies"""
        # Kinetic energy
        KE = 0.5 * (m1 * v**2 + m2 * v**2)
        
        # Potential energy
        PE = -self.G * m1 * m2 / r
        
        return KE, PE

def create_orbit_plot(body1_x, body1_y, body2_x, body2_y, params, t):
    """Create interactive orbital plot"""
    
    fig = make_subplots(rows=2, cols=2,
                       specs=[[{"colspan": 2}, None],
                             [{"type": "scatter"}, {"type": "scatter"}]],
                       subplot_titles=('Orbital Motion',
                                     'Distance from Center', 
                                     'Angular Position'))
    
    # Add body trajectories
    fig.add_trace(
        go.Scatter(x=body1_x, y=body1_y,
                  mode='lines+markers',
                  name=params['body1_name'],
                  line=dict(color='blue'),
                  marker=dict(size=10)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=body2_x, y=body2_y,
                  mode='lines+markers',
                  name=params['body2_name'],
                  line=dict(color='red'),
                  marker=dict(size=8)),
        row=1, col=1
    )
    
    # Add center of mass
    fig.add_trace(
        go.Scatter(x=[0], y=[0],
                  mode='markers',
                  name='Center of Mass',
                  marker=dict(size=5, color='black')),
        row=1, col=1
    )
    
    # Distance from center
    r1 = np.sqrt(body1_x**2 + body1_y**2)
    r2 = np.sqrt(body2_x**2 + body2_y**2)
    
    fig.add_trace(
        go.Scatter(x=t, y=r1, name=f"{params['body1_name']} Distance",
                  line=dict(color='blue')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=t, y=r2, name=f"{params['body2_name']} Distance",
                  line=dict(color='red')),
        row=2, col=1
    )
    
    # Angular position
    theta1 = np.arctan2(body1_y, body1_x)
    theta2 = np.arctan2(body2_y, body2_x)
    
    fig.add_trace(
        go.Scatter(x=t, y=theta1, name=f"{params['body1_name']} Angle",
                  line=dict(color='blue')),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=t, y=theta2, name=f"{params['body2_name']} Angle",
                  line=dict(color='red')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Two-Body System: {params['body1_name']}-{params['body2_name']}",
        hovermode='closest'
    )
    
    fig.update_xaxes(title_text="X Position (km)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (km)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Distance (km)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Angle (rad)", row=2, col=2)
    
    return fig

def main():
    st.set_page_config(page_title="Two-Body Orbital Simulator", layout="wide")
    
    st.title("Two-Body Orbital Motion Simulator")
    st.write("""
    Explore the classical two-body problem with this interactive simulator.
    Select from predefined scenarios or create your own system!
    """)
    
    # Predefined scenarios
    scenarios = {
        'Earth-Moon': {
            'body1_name': 'Earth',
            'body2_name': 'Moon',
            'm1': 5.972e24,
            'm2': 7.342e22,
            'r': 384400e3,
            'description': 'Earth-Moon system'
        },
        'Sun-Earth': {
            'body1_name': 'Sun',
            'body2_name': 'Earth',
            'm1': 1.989e30,
            'm2': 5.972e24,
            'r': 149.6e9,
            'description': 'Sun-Earth system'
        },
        'Custom': {
            'body1_name': 'Body 1',
            'body2_name': 'Body 2',
            'm1': 1.0e24,
            'm2': 1.0e22,
            'r': 1.0e5,
            'description': 'Custom two-body system'
        }
    }
    
    # Sidebar configuration
    st.sidebar.header("System Configuration")
    
    # Scenario selection
    scenario_name = st.sidebar.selectbox(
        "Select Scenario",
        list(scenarios.keys())
    )
    
    scenario = scenarios[scenario_name]
    st.sidebar.markdown(f"**Description:** {scenario['description']}")
    
    # Custom parameters if selected
    if scenario_name == 'Custom':
        st.sidebar.subheader("Body Parameters")
        scenario['body1_name'] = st.sidebar.text_input("Body 1 Name", "Body 1")
        scenario['body2_name'] = st.sidebar.text_input("Body 2 Name", "Body 2")
        scenario['m1'] = st.sidebar.number_input("Mass 1 (kg)", value=1.0e24, format="%.2e")
        scenario['m2'] = st.sidebar.number_input("Mass 2 (kg)", value=1.0e22, format="%.2e")
        scenario['r'] = st.sidebar.number_input("Separation (m)", value=1.0e5, format="%.2e")
    
    # Time settings
    st.sidebar.subheader("Simulation Settings")
    duration = st.sidebar.number_input("Duration (days)", value=30, min_value=1, max_value=365)
    n_points = st.sidebar.slider("Number of Points", 100, 1000, 500)
    
    # Create system and calculate
    system = TwoBodySystem()
    
    if st.sidebar.button("Run Simulation"):
        # Calculate orbital parameters
        force, period, velocity = system.calculate_orbital_parameters(
            scenario['m1'], scenario['m2'], scenario['r']
        )
        
        # Generate time points
        t = np.linspace(0, duration*86400, n_points)
        
        # Calculate positions
        x1, y1, x2, y2 = system.calculate_positions(
            scenario['m1'], scenario['m2'], scenario['r'], t
        )
        
        # Calculate energies
        KE, PE = system.calculate_energies(
            scenario['m1'], scenario['m2'], scenario['r'], velocity
        )
        
        # Create plots
        fig = create_orbit_plot(x1, y1, x2, y2, scenario, t)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display system information
        st.write("### System Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Orbital Period", f"{period/86400:.2f} days")
        with col2:
            st.metric("Orbital Velocity", f"{velocity/1000:.2f} km/s")
        with col3:
            st.metric("Gravitational Force", f"{force:.2e} N")
        
        # Energy information
        st.write("### Energy Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Kinetic Energy", f"{KE:.2e} J")
        with col2:
            st.metric("Potential Energy", f"{PE:.2e} J")
        with col3:
            st.metric("Total Energy", f"{KE + PE:.2e} J")

if __name__ == "__main__":
    main()
```
