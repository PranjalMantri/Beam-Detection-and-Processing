import re

def parse_measurement(text):
    # Remove any extra whitespace and convert to lowercase
    text = ' '.join(text.lower().split())

    # Replace underscores with hyphens
    text = text.replace("_", "-")
    
    # Define regex patterns for measurements and fractions
    measurement_pattern = r'(\d+(?:\.\d+)?)\s*(\'|ft|feet|"|in|inches)(?:\s*-?\s*(\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*("|in|inches))?'
    fraction_pattern = r'(\d+)\s*/\s*(\d+)'
    
    # Find all measurements in the text using the regex pattern
    measurements = re.findall(measurement_pattern, text)
    
    total_inches = 0
    
    # Process each measurement found
    for measurement in measurements:
        value, unit, inches_value, inches_unit = measurement
        
        # Convert feet to inches
        if unit in ["'", "ft", "feet"]:
            total_inches += float(value) * 12
        # Convert inches to inches
        elif unit in ['"', "in", "inches"]:
            total_inches += float(value)
        
        # If there is an additional inches_value (fractional part)
        if inches_value:
            # Check if inches_value contains a fraction
            fraction_match = re.search(fraction_pattern, inches_value)
            if fraction_match:
                # Extract whole number part (if any) and fraction
                whole_part = re.sub(fraction_pattern, '', inches_value).strip()
                whole = float(whole_part) if whole_part else 0
                numerator = int(fraction_match.group(1))
                denominator = int(fraction_match.group(2))
                # Add the whole part and the fraction to total_inches
                total_inches += whole + numerator / denominator
            else:
                # If there's no fraction, add the value directly
                total_inches += float(inches_value)
    
    return total_inches
