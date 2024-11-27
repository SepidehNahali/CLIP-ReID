import xml.etree.ElementTree as ET
import os

def load_vehicle_features(label_file, color_file, type_file, camera_file):
    vehicle_features = {}

    # Load color mappings
    color_map = {}
    with open(color_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                color_id, color_name = parts
                color_map[color_id] = color_name

    # Load type mappings
    type_map = {}
    with open(type_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                type_id, type_name = parts
                type_map[type_id] = type_name

    # Load camera mappings
    camera_map = {}
    with open(camera_file, 'r') as f:
        camera_ids = f.read().strip().split('\t')  # Split by tab
        for camera_id in camera_ids:
            camera_map[camera_id] = f"Camera-{camera_id}"

    print("Color Map:", color_map)
    print("Type Map:", type_map)
    print("Camera Map:", camera_map)

    # Parse the XML file
    try:
        tree = ET.parse(label_file)
        root = tree.getroot()  # <TrainingImages>
        items = root.find("Items")  # Find <Items> within the root

        for item in items.findall("Item"):
            vehicle_id = item.get("vehicleID")
            color_id = item.get("colorID")
            type_id = item.get("typeID")
            camera_id = item.get("cameraID")

            vehicle_features[vehicle_id] = {
                "color": color_map.get(color_id, "Unknown"),
                "type": type_map.get(type_id, "Unknown"),
                "camera_id": camera_map.get(camera_id, f"Unknown-{camera_id}")
            }

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        raise

    return vehicle_features
