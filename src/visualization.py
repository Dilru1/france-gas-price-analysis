import folium
import requests
import os
import json
import pandas as pd

def get_france_geojson():
    """
    Downloads the simplified GeoJSON for French departments.
    Source: gregoiredavid/france-geojson
    """
    # Matches your working URL
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
    save_path = "data/raw/departements-simplifiee.geojson"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download if missing (Caches it so you don't download every time you run)
    if not os.path.exists(save_path):
        print(f"Downloading GeoJSON to {save_path}...")
        resp = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(resp.content)
            
    # Return the JSON object (Dictionary) as Folium expects
    with open(save_path, 'r') as f:
        return json.load(f)

def plot_department_map(pdf_data, gas_name, metric_col="avg_index"):
    """
    Generates a Folium Map for a specific gas type.
    
    Args:
        pdf_data: Pandas DataFrame containing 'dept' and the metric column.
        gas_name: Name of the gas (for the title).
        metric_col: Name of the column with the values (default='avg_index').
    """
    # 1. Get Geometry (The Dictionary)
    geo_data = get_france_geojson()
    
    # 2. Initialize Map
    # Using your coordinates and zoom
    m = folium.Map(location=[46.2276, 2.2137], zoom_start=6, width=800, height=600)
    
    # 3. Create Choropleth Layer
    folium.Choropleth(
        geo_data=geo_data,
        name="choropleth",
        data=pdf_data,
        columns=["dept", metric_col],  # Matches your DataFrame columns
        key_on="feature.properties.code",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"{gas_name} Price Index ({metric_col})"
    ).add_to(m)

    # 4. Add Title (Optional, keeps it professional)
    title_html = f'''
             <h3 align="center" style="font-size:16px"><b>{gas_name} Price Index Heatmap</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))

    return m