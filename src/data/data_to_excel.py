from openpyxl import Workbook

def create_excel_file(data, filename='column_data.xlsx'):
    """
    Creates an Excel file from a list of tuples.
    
    Args:
    - data (list of tuples): The data to write to the Excel file. Each tuple should contain two elements.
    - filename (str): The name of the output Excel file (default is 'output.xlsx').
    
    Returns:
    - None
    """
    # Create a workbook and select the active worksheet
    wb = Workbook()
    ws = wb.active

    # Add headers to the worksheet (optional)
    ws.append(['Category', 'Length (in)'])

    # Insert data into the worksheet
    for entry in data:
        ws.append(entry)

    # Save the workbook
    wb.save(filename)

    print(f"Excel file '{filename}' created successfully.")

# # Example usage
# data = [('C', 10.78700049405221), ('B', 125.46984785187045), ('C', 14.477290136754283), ('B', 268.2556701810352)]
# create_excel_file(data, 'data_output.xlsx')
