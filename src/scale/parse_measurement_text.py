import re

def parse_measurement(text):
    # Remove any extra whitespace and convert to lowercase
    text = ' '.join(text.lower().split())

    if "_" in text:
        text.replace("_", "-")
    
    # Define regex patterns
    measurement_pattern = r'(\d+(?:\.\d+)?)\s*(\'|ft|feet|"|in|inches)(?:\s*-?\s*(\d+(?:\.\d+)?(?:\s+\d+/\d+)?)\s*("|in|inches))?'
    fraction_pattern = r'(\d+)\s*/\s*(\d+)'
    
    # Find all measurements in the text
    measurements = re.findall(measurement_pattern, text)
    
    total_inches = 0
    
    for measurement in measurements:
        value, unit, inches_value, inches_unit = measurement
        
        if unit in ["'", "ft", "feet"]:
            total_inches += float(value) * 12
        elif unit in ['"', "in", "inches"]:
            total_inches += float(value)
        
        if inches_value:
            fraction_match = re.search(fraction_pattern, inches_value)
            if fraction_match:
                whole_part = re.sub(fraction_pattern, '', inches_value).strip()
                whole = float(whole_part) if whole_part else 0
                numerator = int(fraction_match.group(1))
                denominator = int(fraction_match.group(2))
                total_inches += whole + numerator / denominator
            else:
                total_inches += float(inches_value)
    
    return total_inches
