import re

def parse_text(text):
    # Check if "EXTRA" is in the text
    if "EXTRA" in text:
        return "No valid bar found"

    # Pattern to match bar name (alphabet) adjacent to 'Q' or surrounded by '(' or ')'
    pattern = r'(\d+)-(\d+)\s*(Q*\(*([A-Z])\)*)'

    # Find all matches in the text
    matches = re.findall(pattern, text)

    if not matches:
        return "No valid bar found"

    bar_info = []
    bar_names = set()
    for match in matches:
        quantity, measurement, full_match, bar_name = match
        if bar_name != 'Q':
            bar_info.append({
                'bar_name': bar_name,
                'quantity': quantity,
                'measurement': measurement
            })
            bar_names.add(bar_name)

    if not bar_info:
        return "No valid bar found"

    # Check for the presence of other letters besides Q and bar names
    all_letters = set(re.findall(r'[A-Z]', text))
    other_letters = all_letters - set('Q') - bar_names


    if len(other_letters) <= 0:
        bar_type = "THROUGH"
    else:
        bar_type = "EXTRA"


    # Concatenate bar names with a '+' sign
    concatenated_bar_names = "+".join(bar['bar_name'] for bar in bar_info)
    quantities = "+".join(bar['quantity'] for bar in bar_info)
    measurements = "+".join(bar['measurement'] for bar in bar_info)

    return f"{bar_type} Bar(s): {concatenated_bar_names}, Quantity: {quantities}, Measurement: {measurements}"


if __name__ == "__main__":
    # Test the function
    test_cases = [
        "2-16Q(A)",
        "(A)+4-20Q(B)",
        "2-20(C)+2-20Q(D)",
        "(C)+(D)+2-25Q(E)",
        "2-10EXTRA",
        "108",
        "4\"c/c12NOS",
        "2-16(A)+2-20Q",
        "2-20(C)+2-20(D)+3-25Q(E)"
    ]

    for case in test_cases:
        result = parse_text(case)
        print(f"Input: {case}")
        print(f"Output: {result}")
        print()
