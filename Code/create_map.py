import folium

# 1. Create a map centered at some location (latitude, longitude)
m = folium.Map(location=[37.7749, -122.4194], zoom_start=12)

# 2. Add a marker (optional)
folium.Marker([37.7749, -122.4194], popup="San Francisco").add_to(m)

# 3. Save the map to an HTML file
m.save("map.html")
