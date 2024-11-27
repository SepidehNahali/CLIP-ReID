import xml.etree.ElementTree as ET
import codecs  # For handling multi-byte encodings like gb2312

def load_vehicle_features(label_file, color_file, type_file, camera_file):
    vehicle_features = {}

    # Load color, type, and camera mappings
    color_map = {}
    type_map = {}
    camera_map = {}

    with open(color_file, 'r') as f:
        for line in f:
            color_id, color_name = line.strip().split(' ', 1)
            color_map[color_id] = color_name

    with open(type_file, 'r') as f:
        for line in f:
            type_id, type_name = line.strip().split(' ', 1)
            type_map[type_id] = type_name

    with open(camera_file, 'r') as f:
        for line in f:
            camera_id = line.strip()
            camera_map[camera_id] = f"Camera-{camera_id}"

    # Read and decode the XML file with gb2312 encoding
    try:
        with codecs.open(label_file, 'r', encoding='gb2312') as f:
            content = f.read()

        # Parse the XML content
        root = ET.fromstring(content)

        for item in root.findall('Items/Item'):
            vehicle_id = item.get('vehicleID')
            color_id = item.get('colorID')
            type_id = item.get('typeID')
            camera_id = item.get('cameraID')

            vehicle_features[vehicle_id] = {
                "color": color_map.get(color_id, "Unknown"),
                "type": type_map.get(type_id, "Unknown"),
                "camera_id": camera_map.get(camera_id, camera_id)
            }

    except ValueError as e:
        print(f"Encoding issue with the XML file: {e}")
        raise
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        raise

    return vehicle_features
