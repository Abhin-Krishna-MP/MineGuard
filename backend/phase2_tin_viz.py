import geemap
import numpy as np
import plotly.graph_objects as go
import os

def generate_tin_visualization(depth_image, roi_geometry, illegal_area_m2, output_path="mineguard_3d_tin.html"):
    """
    Phase 2 Module: Converts GEE Depth Raster into a 3D TIN visualization.
    """
    print(f"⏳ Generating 3D TIN Model...")

    try:
        # 1. DOWNLOAD DATA
        scale = 30 
        if illegal_area_m2 > 1000000: scale = 60 
            
        depth_array = geemap.ee_to_numpy(depth_image.unmask(0), region=roi_geometry, scale=scale)
        
        if depth_array is None:
            print("⚠️ No valid depth data found for 3D modeling.")
            return None

        # Fix 3D array shape if necessary
        if depth_array.ndim == 3:
            depth_array = np.squeeze(depth_array)

        if np.max(depth_array) == 0:
            print("⚠️ Depth array is empty.")
            return None

        # 2. PREPARE 3D COORDINATES
        rows, cols = depth_array.shape
        x = np.linspace(0, cols * scale, cols)
        y = np.linspace(0, rows * scale, rows)
        X, Y = np.meshgrid(x, y)
        Z = -depth_array # Negative Z for pits

        # 3. CREATE 3D PLOT
        fig = go.Figure(data=[go.Surface(
            z=Z, x=X, y=Y,
            colorscale='Earth', 
            cmin=np.min(Z), cmax=0,
            contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
            lighting=dict(roughness=0.9, specular=0.1, ambient=0.5)
        )])

        fig.update_layout(
            title=f'<b>MineGuard 3D Forensics Model</b>',
            autosize=True,
            scene=dict(
                xaxis_title='East (m)', yaxis_title='North (m)', zaxis_title='Depth (m)',
                aspectratio=dict(x=1, y=1, z=0.4), 
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            template="plotly_dark"
        )

        # 4. SAVE OUTPUT
        fig.write_html(output_path)
        print(f"🗺️  3D Visualization Saved: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"⚠️ 3D Generation Failed: {e}")
        return None