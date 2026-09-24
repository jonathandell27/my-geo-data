import requests
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# General Settings
# =============================================================================

BASE_URL = "https://gisn.tel-aviv.gov.il/arcgis/rest/services/WM/IView2WM/MapServer" 
OUTPUT_DIR = Path("polygon_intersects_html")
CSV_PATH = OUTPUT_DIR / "intersecting_features_summary.csv"
LOG_PATH = OUTPUT_DIR / "run_log.txt"

# -----------------------------------------------------------------------------
# Geojson for exercise 4
# -----------------------------------------------------------------------------
INPUT_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [34.780133, 32.081907],
                        [34.780133, 32.081283],
                        [34.781110, 32.081283],
                        [34.781110, 32.081907],
                        [34.780133, 32.081907],
                    ]
                ],
            },
            "properties": {"ID": 0},
        }
    ],
}

# Conversion to Shapely for spatial analysis
USER_POLY = shape(INPUT_GEOJSON["features"][0]["geometry"])


# =============================================================================
# Tee ​​writes to the screen and a log file
# =============================================================================
class Tee:
    """
    כותב בו-זמנית למסך ולקובץ לוג.
    מוסיף חותמת זמן בתחילת כל שורה חדשה.
    """

    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout
        self._at_line_start = True 

    def _timestamp(self):
        return datetime.now().strftime("%H:%M:%S")

    def write(self, message):
        parts = message.split("\n")
        output = []

        for i, part in enumerate(parts):
            if i > 0:
                output.append("\n")
                self._at_line_start = True

            if part == "":
                continue

            if self._at_line_start:
                ts = f"[{self._timestamp()}] "
                output.append(ts + part)
                self._at_line_start = False
            else:
                output.append(part)

        text = "".join(output)
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# =============================================================================
# Polygon and metadata 
# =============================================================================
def get_polygon_layers():
    """
    קורא את Metadata של ה-MapServer ומחזיר רק Feature Layers פוליגונליים.

    פנייה: GET {BASE_URL}?f=pjson
    סינון: type == Feature Layer  וגם  geometryType == esriGeometryPolygon
    """
    response = requests.get(BASE_URL, params={"f": "pjson"}, timeout=30)
    response.raise_for_status()

    all_layers = response.json()["layers"]
    polygon_layers = []

    for layer in all_layers:
        if layer.get("type") != "Feature Layer":
            continue
        if layer.get("geometryType") != "esriGeometryPolygon":
            continue
        polygon_layers.append({
            "id": layer["id"],
            "name": layer["name"],
        })

    return polygon_layers


# =============================================================================
# Overlapping featers + full/partial classification
# =============================================================================
def build_arcgis_geometry_param():
    """ממיר את הפוליגון לפורמט geometry של ArcGIS REST."""
    coordinates = INPUT_GEOJSON["features"][0]["geometry"]["coordinates"]
    return json.dumps({
        "rings": coordinates,
        "spatialReference": {"wkid": 4326},
    })


def fetch_intersecting_features(layer_id: int) -> gpd.GeoDataFrame | None:
    """
    שולף ישויות שחופפות את הפוליגון (Intersects).

    זו השאילתה הבסיסית לשרת – מחזירה גם הכלה מלאה וגם חפיפה חלקית.
    את הסיווג המדויק עושים אחר כך ב-Shapely.
    """
    params = {
        "geometry": build_arcgis_geometry_param(),
        "geometryType": "esriGeometryPolygon",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }

    try:
        url = f"{BASE_URL}/{layer_id}/query"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "error" in data or not data.get("features"):
            return None

        return gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

    except Exception as e:
        print(f"    [שגיאת שאילתה] {e}")
        return None


def classify_intersection(geom) -> str:
    """
    מסווג את סוג החפיפה בין הישות לפוליגון הנתון.

    מחזיר:
      "full"    – הכלה מלאה (הישות מכילה את הפוליגון, או להפך)
      "partial" – חפיפה חלקית בלבד
    """
    if geom is None or geom.is_empty:
        return "partial"
    try:
        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.contains(USER_POLY) or USER_POLY.within(geom):
            return "full"
        return "partial"
    except Exception:
        return "partial"


def get_feature_id(row, gdf):
    """מחלץ Feature ID (OBJECTID / OID / FID...')."""
    for col in ["OBJECTID", "objectid", "OID", "oid", "FID", "fid", "id", "ID"]:
        if col in gdf.columns and pd.notna(row.get(col)):
            return row[col]
    return row.name


def split_to_geojson_lists(gdf):
    """
    מפצל את ה-GeoDataFrame לשתי רשימות GeoJSON:
      - features_full
      - features_partial
    """
    features_full = []
    features_partial = []

    for _, row in gdf.iterrows():
        try:
            geom_json = json.loads(
                gpd.GeoSeries([row.geometry]).to_json()
            )["features"][0]["geometry"]

            props = {}
            for key, val in row.drop(
                labels=["geometry", "intersect_type"], errors="ignore"
            ).items():
                if pd.notna(val):
                    if hasattr(val, "item"):
                        val = val.item()
                    props[str(key)] = val

            props["intersect_type"] = row["intersect_type"]

            feature = {
                "type": "Feature",
                "geometry": geom_json,
                "properties": props,
            }

            if row["intersect_type"] == "full":
                features_full.append(feature)
            else:
                features_partial.append(feature)

        except Exception:
            continue

    return features_full, features_partial


# =============================================================================
# HTML Map – Distinguishing between full and partial overlap
# =============================================================================
def create_leaflet_html(
    features_full, features_partial, layer_id, layer_name, output_path
):
    """
    מפת Leaflet:
      - אדום  = הפוליגון הנתון (GeoJSON)
      - ירוק  = חפיפה מלאה (הכלה)
      - כחול  = חפיפה חלקית
    המקרא מציג רק סוגים שקיימים בפועל.
    """
    user_json = json.dumps(INPUT_GEOJSON)
    full_json = json.dumps(
        {"type": "FeatureCollection", "features": features_full}
    )
    partial_json = json.dumps(
        {"type": "FeatureCollection", "features": features_partial}
    )
    legend_name = layer_name.replace('"', "'")

    has_full = len(features_full) > 0
    has_partial = len(features_partial) > 0

    # בניית המקרא באופן דינמי
    legend_items = [
        '<div class="legend-item"><span class="box user-poly"></span> פוליגון בgeojson</div>'
    ]
    if has_full:
        legend_items.append(
            '<div class="legend-item"><span class="box full-poly"></span> חפיפה מלאה</div>'
        )
    if has_partial:
        legend_items.append(
            '<div class="legend-item"><span class="box partial-poly"></span> חפיפה חלקית</div>'
        )
    legend_html = "\n        ".join(legend_items)

    # שכבות JS
    if has_full:
        full_js = f"""
        const fullLayer = L.geoJSON({full_json}, {{
            style: {{
                color: '#1e8449', weight: 2.5,
                fillColor: '#27ae60', fillOpacity: 0.70
            }},
            onEachFeature: (f, layer) => layer.bindPopup(popupContent(f, 'חפיפה מלאה'))
        }}).addTo(map);
        """
    else:
        full_js = "const fullLayer = null;"

    if has_partial:
        partial_js = f"""
        const partialLayer = L.geoJSON({partial_json}, {{
            style: {{
                color: '#1a5276', weight: 1.5,
                fillColor: '#3498db', fillOpacity: 0.40
            }},
            onEachFeature: (f, layer) => layer.bindPopup(popupContent(f, 'חפיפה חלקית'))
        }}).addTo(map);
        """
    else:
        partial_js = "const partialLayer = null;"

    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ID {layer_id} | {layer_name}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: "Segoe UI", Arial, sans-serif; }}
        #map {{ width: 100%; height: 100vh; }}
        .info-box {{
            position: absolute; top: 12px; right: 12px; z-index: 1000;
            background: rgba(255,255,255,0.95); padding: 14px 16px;
            border-radius: 10px; box-shadow: 0 2px 14px rgba(0,0,0,0.3);
            max-width: 320px; font-size: 13.5px; line-height: 1.45;
        }}
        .info-box h3 {{ margin: 0 0 8px 0; font-size: 15px; }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-item span.box {{
            display: inline-block; width: 16px; height: 16px;
            margin-left: 8px; border-radius: 3px; flex-shrink: 0;
        }}
        .user-poly {{ background: #e74c3c; border: 2px solid #7b241c; }}
        .full-poly {{ background: #27ae60; border: 2px solid #1e8449; }}
        .partial-poly {{ background: #3498db; border: 2px solid #1a5276; opacity: 0.7; }}
        .note {{ margin-top: 8px; font-size: 12px; color: #555;
                 border-top: 1px solid #ddd; padding-top: 6px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box">
        <h3>ID {layer_id} — {legend_name}</h3>
        {legend_html}
        <div class="note">לחיצה על פוליגון מציגה את השדות</div>
    </div>
    <script>
        const map = L.map('map').setView([32.0816, 34.7806], 17);

        L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
            {{ maxZoom: 19, attribution: 'Esri World Imagery' }}
        ).addTo(map);

        function popupContent(feature, label) {{
            const props = feature.properties || {{}};
            let html = '<b>' + label + '</b><hr style="margin:6px 0">';
            for (const key in props) {{
                if (key === 'intersect_type') continue;
                if (props[key] !== null && props[key] !== undefined)
                    html += '<b>' + key + ':</b> ' + props[key] + '<br>';
            }}
            return html;
        }}

        {partial_js}
        {full_js}

        const userLayer = L.geoJSON({user_json}, {{
            style: {{
                color: '#7b241c', weight: 3,
                fillColor: '#e74c3c', fillOpacity: 0.55
            }}
        }}).addTo(map);

        const layersForBounds = [userLayer];
        if (fullLayer) layersForBounds.push(fullLayer);
        if (partialLayer) layersForBounds.push(partialLayer);

        const group = new L.featureGroup(layersForBounds);
        if (group.getLayers().length)
            map.fitBounds(group.getBounds().pad(0.2));
    </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


# =============================================================================
# The main run
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    log = Tee(LOG_PATH)
    sys.stdout = log

    try:
        print(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output folder: {OUTPUT_DIR.resolve()}\n")

        # ----- שלב 1+2 -----
        print("=" * 60)
        print("Step 1: Reading MapServer metadata")
        print("Step 2: Locating polygon layers")
        print("=" * 60)

        polygon_layers = get_polygon_layers()
        print(f"find {len(polygon_layers)} polygon layers \n")

        # ----- שלב 3 -----
        print("=" * 60)
        print("Step 3: Search for overlapping features + full/partial classification")
        print("=" * 60)

        csv_rows = []
        saved_maps = []

        for i, layer in enumerate(polygon_layers, 1):
            layer_id = layer["id"]
            layer_name = layer["name"]
            print(f"[{i:02d}/{len(polygon_layers)}] layer {layer_id} – {layer_name}")

            # Intersects
            gdf = fetch_intersecting_features(layer_id)
            if gdf is None or gdf.empty:
                print("         No overlap")
                continue

            # Classification of each features: full / partial
            gdf["intersect_type"] = gdf.geometry.apply(classify_intersection)

            n_full = (gdf["intersect_type"] == "full").sum()
            n_partial = (gdf["intersect_type"] == "partial").sum()
            print(f"         full={n_full} | partial={n_partial}")

            # Split to GeoJSON
            features_full, features_partial = split_to_geojson_lists(gdf)

            # מפת HTML
            safe_name = "".join(
                c if c.isalnum() or c in " -_" else "_" for c in layer_name
            )[:55]
            html_path = OUTPUT_DIR / f"{layer_id:04d}_{safe_name}.html"
            create_leaflet_html(
                features_full, features_partial,
                layer_id, layer_name, html_path
            )
            saved_maps.append(html_path.name)

            # CSV rows
            for _, row in gdf.iterrows():
                csv_rows.append({
                    "Layer Name": layer_name,
                    "Layer ID": layer_id,
                    "Feature ID": get_feature_id(row, gdf),
                    "Intersect Type": row["intersect_type"],  # full / partial
                })

            time.sleep(0.12)

        # ----- Step 4: CSV -----
        print("\n" + "=" * 60)
        print("Step 4: Creating a CSV file")
        print("=" * 60)

        if csv_rows:
            df = pd.DataFrame(csv_rows)
            df = df.sort_values(["Layer ID", "Intersect Type", "Feature ID"])
            df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

            print(f"saved: {CSV_PATH}")
            print(f"Total records:: {len(df)}")
            print(f"full overlap:    {(df['Intersect Type'] == 'full').sum()}")
            print(f"partial overlap: {(df['Intersect Type'] == 'partial').sum()}")
        else:
            print("No overlapping features were found.")

        # סיכום
        print("\n" + "=" * 60)
        print("End")
        print("=" * 60)
        print(f"Maps HTML: {len(saved_maps)}")
        print(f"CSV:       {CSV_PATH.name}")
        print(f"Log:       {LOG_PATH.name}")
        print(f"folder:    {OUTPUT_DIR.resolve()}")
        print(f"end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        sys.stdout = log.stdout
        log.close()
        print(f"\n✓ Log file was save: {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()