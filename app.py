"""
Subsurface Ocean Temperature Reconstruction — Prototype UI
North Indian Ocean (5N-30N, 45E-105E)

This is a FRONTEND-ONLY prototype. All predictions come from
model_interface.py, which currently contains mock/placeholder logic.
Swap that module for real model inference later — the app itself
will not need to change.
"""

import datetime

import folium
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from map_utils import field_to_overlay_image, image_to_data_url, make_colormap_legend
from model_interface import (
    GRID_RESOLUTION_DEG,
    REGION_BOUNDS,
    STANDARD_DEPTHS,
    predict_region_field,
    predict_region_profile_stats,
    predict_temperature_profile,
)

st.set_page_config(page_title="Subsurface Ocean Temperature Reconstruction", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "selected_lat": None,
    "selected_lon": None,
    "selected_region": None,
    "prev_mode": None,
    "last_profile": None,
    "last_region_stats": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def in_bounds(lat, lon):
    return (
        REGION_BOUNDS["lat_min"] <= lat <= REGION_BOUNDS["lat_max"]
        and REGION_BOUNDS["lon_min"] <= lon <= REGION_BOUNDS["lon_max"]
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Subsurface Ocean Temperature Reconstruction")
st.caption(
    "Satellite Embedding–Based Deep Learning Framework · North Indian Ocean "
    f"({REGION_BOUNDS['lat_min']}°N–{REGION_BOUNDS['lat_max']}°N, "
    f"{REGION_BOUNDS['lon_min']}°E–{REGION_BOUNDS['lon_max']}°E) · "
    f"{GRID_RESOLUTION_DEG}° resolution, daily"
)

with st.expander("About this prototype", expanded=False):
    st.markdown(
        """
**How to use it:**
- The map always shows the reconstructed temperature field at the chosen
  depth & date, styled as smooth filled contours over the real coastline.
- Switch **Selection mode** in the sidebar to either:
  - **Point** — click anywhere on the map to get a temperature-vs-depth
    profile at that exact location, or
  - **Region (rectangle)** — draw a box on the map (rectangle tool, top-left
    of the map) to get min/mean/max/std statistics and an averaged profile
    over that whole area.

        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Inputs")

selection_mode = st.sidebar.radio("Selection mode", ["Point", "Region (rectangle)"])

if st.session_state.prev_mode != selection_mode:
    st.session_state.selected_lat = None
    st.session_state.selected_lon = None
    st.session_state.selected_region = None
    st.session_state.last_profile = None
    st.session_state.last_region_stats = None
    st.session_state.prev_mode = selection_mode

date = st.sidebar.date_input(
    "Date",
    value=datetime.date(2023, 6, 15),
    min_value=datetime.date(1993, 1, 1),
    max_value=datetime.date.today(),
)

st.sidebar.subheader("Surface observations")
st.sidebar.caption("Used for point-mode predictions")
sst = st.sidebar.number_input("Sea Surface Temperature — SST (°C)", value=28.5, min_value=-2.0, max_value=36.0, step=0.1)
sss = st.sidebar.number_input("Sea Surface Salinity — SSS / so (PSU)", value=35.0, min_value=25.0, max_value=40.0, step=0.1)
ssh = st.sidebar.number_input("Sea Surface Height / SLA (m)", value=0.10, min_value=-1.0, max_value=1.0, step=0.01)
uo = st.sidebar.number_input("Surface current — uo (m/s)", value=0.20, min_value=-2.0, max_value=2.0, step=0.01)
vo = st.sidebar.number_input("Surface current — vo (m/s)", value=-0.10, min_value=-2.0, max_value=2.0, step=0.01)

st.sidebar.subheader("Depth")
depth = st.sidebar.slider("Depth level (m) — controls the map layer", min_value=0, max_value=1000, value=0, step=5)

st.sidebar.divider()
if selection_mode == "Point":
    if st.session_state.selected_lat is not None:
        st.sidebar.success(f"Point: {st.session_state.selected_lat:.3f}°, {st.session_state.selected_lon:.3f}°")
        if st.sidebar.button("Clear point"):
            st.session_state.selected_lat = None
            st.session_state.selected_lon = None
            st.session_state.last_profile = None
            st.rerun()
    else:
        st.sidebar.info("Click the map to pick a location.")
else:
    if st.session_state.selected_region is not None:
        r = st.session_state.selected_region
        st.sidebar.success(f"Region: {r['lat_min']:.2f}–{r['lat_max']:.2f}°N, {r['lon_min']:.2f}–{r['lon_max']:.2f}°E")
        if st.sidebar.button("Clear region"):
            st.session_state.selected_region = None
            st.session_state.last_region_stats = None
            st.rerun()
    else:
        st.sidebar.info("Use the rectangle tool (top-left of map) to draw a region.")

# ---------------------------------------------------------------------------
# Build the temperature field for the map (always visible, all modes)
# ---------------------------------------------------------------------------
lat_grid = np.arange(REGION_BOUNDS["lat_min"], REGION_BOUNDS["lat_max"] + 1e-6, GRID_RESOLUTION_DEG)
lon_grid = np.arange(REGION_BOUNDS["lon_min"], REGION_BOUNDS["lon_max"] + 1e-6, GRID_RESOLUTION_DEG)
field = predict_region_field(depth=depth, date=date, lat_grid=lat_grid, lon_grid=lon_grid)
vmin, vmax = float(np.nanmin(field)), float(np.nanmax(field))

overlay_img = field_to_overlay_image(field, vmin, vmax)
overlay_url = image_to_data_url(overlay_img)

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
col_map, col_result = st.columns([1.3, 1], gap="large")

with col_map:
    st.subheader("Reconstructed temperature field")
    st.caption(f"Depth {depth} m · {date.isoformat()} — smooth overlay is illustrative, styled to match reference imagery")

    center_lat = (REGION_BOUNDS["lat_min"] + REGION_BOUNDS["lat_max"]) / 2
    center_lon = (REGION_BOUNDS["lon_min"] + REGION_BOUNDS["lon_max"]) / 2

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="OpenStreetMap")

    folium.raster_layers.ImageOverlay(
        image=overlay_url,
        bounds=[
            [REGION_BOUNDS["lat_min"], REGION_BOUNDS["lon_min"]],
            [REGION_BOUNDS["lat_max"], REGION_BOUNDS["lon_max"]],
        ],
        opacity=0.8,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(fmap)

    make_colormap_legend(vmin, vmax).add_to(fmap)

    folium.Rectangle(
        bounds=[
            [REGION_BOUNDS["lat_min"], REGION_BOUNDS["lon_min"]],
            [REGION_BOUNDS["lat_max"], REGION_BOUNDS["lon_max"]],
        ],
        color="#333333",
        weight=1.5,
        fill=False,
        dash_array="4,4",
        tooltip="Study region",
    ).add_to(fmap)

    return_objs = ["last_clicked"]
    if selection_mode == "Region (rectangle)":
        Draw(
            export=False,
            draw_options={
                "rectangle": {"shapeOptions": {"color": "#d62728", "weight": 2}},
                "polygon": False,
                "circle": False,
                "circlemarker": False,
                "marker": False,
                "polyline": False,
            },
            edit_options={"edit": True, "remove": True},
        ).add_to(fmap)
        return_objs.append("last_active_drawing")

        if st.session_state.selected_region is not None:
            r = st.session_state.selected_region
            folium.Rectangle(
                bounds=[[r["lat_min"], r["lon_min"]], [r["lat_max"], r["lon_max"]]],
                color="#d62728",
                weight=2,
                fill=True,
                fill_opacity=0.06,
            ).add_to(fmap)
    else:
        if st.session_state.selected_lat is not None:
            folium.Marker(
                location=[st.session_state.selected_lat, st.session_state.selected_lon],
                icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
                tooltip="Selected point",
            ).add_to(fmap)

    map_state = st_folium(fmap, height=560, use_container_width=True, key="ocean_map", returned_objects=return_objs)

    # ---- handle interaction ----
    if selection_mode == "Point":
        if map_state and map_state.get("last_clicked"):
            clicked_lat = map_state["last_clicked"]["lat"]
            clicked_lon = map_state["last_clicked"]["lng"]
            if in_bounds(clicked_lat, clicked_lon):
                if (
                    st.session_state.selected_lat != clicked_lat
                    or st.session_state.selected_lon != clicked_lon
                ):
                    st.session_state.selected_lat = clicked_lat
                    st.session_state.selected_lon = clicked_lon
                    st.session_state.last_profile = None
                    st.rerun()
            else:
                st.warning("That point is outside the study region.")
    else:
        drawing = map_state.get("last_active_drawing") if map_state else None
        if drawing and drawing.get("geometry", {}).get("type") == "Polygon":
            coords = drawing["geometry"]["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            new_region = {
                "lat_min": max(min(lats), REGION_BOUNDS["lat_min"]),
                "lat_max": min(max(lats), REGION_BOUNDS["lat_max"]),
                "lon_min": max(min(lons), REGION_BOUNDS["lon_min"]),
                "lon_max": min(max(lons), REGION_BOUNDS["lon_max"]),
            }
            if new_region != st.session_state.selected_region:
                st.session_state.selected_region = new_region
                st.session_state.last_region_stats = None
                st.rerun()

# ---- Results (right column) ------------------------------------------------
with col_result:
    if selection_mode == "Point":
        if st.session_state.selected_lat is not None:
            st.subheader("Temperature profile")
            st.caption(
                f"Lat {st.session_state.selected_lat:.3f}°, Lon {st.session_state.selected_lon:.3f}°  ·  {date.isoformat()}"
            )
            run = st.button("Predict Profile", type="primary", use_container_width=True)

            if run:
                st.session_state.last_profile = predict_temperature_profile(
                    lat=st.session_state.selected_lat,
                    lon=st.session_state.selected_lon,
                    sst=sst, sss=sss, ssh=ssh, uo=uo, vo=vo, date=date,
                )

            if st.session_state.last_profile is not None:
                result = st.session_state.last_profile
                depths_arr = result["depths"]
                temp_arr = result["temperature"]
                nearest_idx = int(np.argmin(np.abs(np.array(depths_arr) - depth)))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=temp_arr, y=depths_arr, mode="lines+markers",
                    line=dict(color="#1f77b4", width=2), marker=dict(size=6), name="Temperature",
                ))
                fig.add_trace(go.Scatter(
                    x=[temp_arr[nearest_idx]], y=[depths_arr[nearest_idx]], mode="markers",
                    marker=dict(size=14, color="red", symbol="star"),
                    name=f"Selected depth ({depths_arr[nearest_idx]:.0f} m)",
                ))
                fig.update_layout(
                    xaxis_title="Temperature (°C)", yaxis_title="Depth (m)",
                    yaxis=dict(autorange="reversed"), height=460,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Values table"):
                    st.dataframe(
                        {"Depth (m)": depths_arr, "Temperature (°C)": [round(t, 3) for t in temp_arr]},
                        use_container_width=True, hide_index=True,
                    )
            else:
                st.info("Set the surface inputs in the sidebar, then click **Predict Profile**.")
        else:
            st.subheader("Temperature profile")
            st.info("Click a point on the map to select a location.")

    else:  # Region mode
        if st.session_state.selected_region is not None:
            r = st.session_state.selected_region
            st.subheader("Region statistics")
            st.caption(
                f"Lat {r['lat_min']:.2f}°–{r['lat_max']:.2f}°, Lon {r['lon_min']:.2f}°–{r['lon_max']:.2f}°  ·  {date.isoformat()}"
            )
            run = st.button("Predict Region Stats", type="primary", use_container_width=True)

            if run:
                st.session_state.last_region_stats = predict_region_profile_stats(
                    lat_min=r["lat_min"], lat_max=r["lat_max"],
                    lon_min=r["lon_min"], lon_max=r["lon_max"], date=date,
                )

            if st.session_state.last_region_stats is not None:
                stats = st.session_state.last_region_stats
                depths_arr = stats["depths"]
                nearest_idx = int(np.argmin(np.abs(np.array(depths_arr) - depth)))

                st.caption(f"{stats['n_points']} grid cells sampled at {depths_arr[nearest_idx]:.0f} m")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean", f"{stats['mean'][nearest_idx]:.2f} °C")
                m2.metric("Min", f"{stats['min'][nearest_idx]:.2f} °C")
                m3.metric("Max", f"{stats['max'][nearest_idx]:.2f} °C")
                m4.metric("Std Dev", f"{stats['std'][nearest_idx]:.2f} °C")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=stats["max"], y=depths_arr, line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=stats["min"], y=depths_arr, fill="tonextx", fillcolor="rgba(31,119,180,0.18)",
                    line=dict(width=0), name="Min–Max range",
                ))
                fig.add_trace(go.Scatter(
                    x=stats["mean"], y=depths_arr, line=dict(color="#1f77b4", width=3), name="Mean",
                ))
                fig.add_trace(go.Scatter(
                    x=[stats["mean"][nearest_idx]], y=[depths_arr[nearest_idx]], mode="markers",
                    marker=dict(size=14, color="red", symbol="star"), name=f"Selected depth ({depths_arr[nearest_idx]:.0f} m)",
                ))
                fig.update_layout(
                    xaxis_title="Temperature (°C)", yaxis_title="Depth (m)",
                    yaxis=dict(autorange="reversed"), height=400,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Values table"):
                    st.dataframe(
                        {
                            "Depth (m)": depths_arr,
                            "Mean (°C)": [round(v, 3) for v in stats["mean"]],
                            "Min (°C)": [round(v, 3) for v in stats["min"]],
                            "Max (°C)": [round(v, 3) for v in stats["max"]],
                            "Std Dev (°C)": [round(v, 3) for v in stats["std"]],
                        },
                        use_container_width=True, hide_index=True,
                    )
            else:
                st.info("Click **Predict Region Stats** to compute statistics for this box.")
        else:
            st.subheader("Region statistics")
            st.info("Draw a rectangle on the map (top-left drawing tool) to select a region.")

st.divider()
# st.caption(
#     "Prototype UI only — reconstruction values are placeholder/mock outputs from `model_interface.py`. "
#     "Replace with the trained satellite-embedding model for real predictions."
# )