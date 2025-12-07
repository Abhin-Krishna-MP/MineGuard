import geemap
import numpy as np
import plotly.graph_objects as go

def generate_tin_visualization(combined_image, roi_geometry, total_area_m2, output_path="mineguard_3d_tin.html"):
    """
    Generates a Raw, True-Scale 3D Terrain Model.
    - Scale: 1:1 (True Physics).
    - Lighting: Default Plotly (Natural shadows).
    - Style: Clean mesh without extra contour lines.
    """
    print("⏳ Generating 3D Forensic Model (Standard True Scale)...")

    try:
        # 1. DOWNLOAD DATA
        scale = 30 
        if total_area_m2 > 1000000: scale = 60 
            
        data_array = geemap.ee_to_numpy(combined_image.unmask(0), region=roi_geometry, scale=scale)
        
        if data_array is None or data_array.ndim != 3:
            print("⚠️ Invalid data for 3D modeling.")
            return None

        depth_grid = data_array[:, :, 0]   # Depth
        status_grid = data_array[:, :, 1]  # 1=Illegal, 2=Legal

        # Trim empty borders
        rows_with_data = np.any(status_grid > 0, axis=1)
        cols_with_data = np.any(status_grid > 0, axis=0)
        
        if not np.any(rows_with_data):
            print("⚠️ Status grid is empty.")
            return None
            
        depth_grid = depth_grid[rows_with_data][:, cols_with_data]
        status_grid = status_grid[rows_with_data][:, cols_with_data]

        # 2. PREPARE MESH
        rows, cols = depth_grid.shape
        x = np.linspace(0, cols * scale, cols)
        y = np.linspace(0, rows * scale, rows)
        X, Y = np.meshgrid(x, y)
        
        # Z = Negative Depth
        Z = -depth_grid 

        # 3. COLORS (Standard Terrain Palette)
        surface_color = status_grid.astype(float)

        colorscale = [
            [0.0, '#1e293b'],           # 0: Background (Slate)
            [0.49, '#1e293b'],
            [0.5, '#8b4513'],           # 1: Illegal -> Standard Earth Brown
            [0.9, '#a52a2a'],           # 1: Illegal -> Reddish Brown
            [1.0, '#22c55e']            # 2: Legal   -> Standard Green
        ]

        # 4. CREATE 3D SURFACE (Clean & Simple)
        fig = go.Figure(data=[go.Surface(
            z=Z, x=X, y=Y,
            surfacecolor=surface_color,
            colorscale=colorscale,
            cmin=0, cmax=2,
            showscale=False,
            # Using Default Lighting (No custom dictionary) for natural terrain look
        )])

        # 5. STYLING
        tin_volume = np.sum(depth_grid) * (scale * scale)
        z_max = np.max(depth_grid)

        fig.update_layout(
            title=dict(
                text=f'<b>3D Terrain Analysis</b><br><sup>Vol: {tin_volume:,.0f} m³ | Max Depth: {z_max:.1f}m</sup>',
                font=dict(color="#e2e8f0", size=18),
                y=0.9
            ),
            autosize=True,
            scene=dict(
                xaxis=dict(title=dict(text='East (m)', font=dict(color="gray")), showbackground=False, gridcolor="#334155"),
                yaxis=dict(title=dict(text='North (m)', font=dict(color="gray")), showbackground=False, gridcolor="#334155"),
                zaxis=dict(title=dict(text='Depth (m)', font=dict(color="gray")), showbackground=True, backgroundcolor="rgb(15,15,20)", gridcolor="#334155"),
                
                # *** 1:1 TRUE SCALE ***
                aspectmode='data', 
                
                camera=dict(
                    eye=dict(x=1.4, y=1.4, z=1.2), # Standard isometric view
                    center=dict(x=0, y=0, z=-0.1)
                ),
                bgcolor="rgba(0,0,0,0)"
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            paper_bgcolor="#0f172a",
        )

        fig.write_html(output_path)
        print(f"🗺️  3D Visualization Saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"⚠️ 3D Generation Failed: {e}")
        return None