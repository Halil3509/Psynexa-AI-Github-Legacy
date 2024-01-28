import numpy as np


def change_ratios(values_dict, value, name, type = "inc"):
    """Add ratio

    Args:
        values_dict (_type_): _description_
        value (_type_): inc or dec adding ratio value
        name (_type_): _description_
        type (str, optional): _description_. Defaults to "inc".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    print(f"Changing ratios process is starting for {name}...")
    
    old_value = values_dict[name]
    if type == "inc":
        new_value = old_value + value

    elif type == "dec":  
        new_value = old_value - value
        
        if new_value < 0:
            new_value = 0
            print("New value is less than zero : ", new_value)
            
    else:
        raise ValueError("type parameter must be inc or dec!")
    
    coff_new = 100 / (100 + new_value)
    coff_old = 100 / (100 + old_value)
    
    average_value = np.round((old_value * coff_old + new_value * coff_new) / (coff_old + coff_new), 2) 

    values_dict[name] = average_value

    total = sum(values_dict.values())
    # Step 2 and 3: Divide by total and multiply by 100
    percentages = {key: np.round((value / total) * 100, 2) for key, value in values_dict.items()}

    print(f"Changing ratios process finished for {name}.")
    return percentages