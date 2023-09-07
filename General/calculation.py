import numpy as np

def change_ratios(values_dict, value, name, type = "inc"):
    old_value = values_dict[name]
    
    if type == "inc":
        new_value = (old_value + value)
        coff = old_value / new_value
        values_dict[name] = new_value
        
    elif type == "dec":  
        new_value = old_value - value
        
        if new_value < 0:
            coff = 1
            print("New value is less than zero : ", new_value)
        else:
            coff = old_value / new_value
            values_dict[name] = new_value
    else:
        raise ValueError("type parameter must be inc or dec!")

    for column in values_dict.keys():
        if column != name:
            values_dict[column] = values_dict[column] * coff

    
    total = sum(values_dict.values())
    # Step 2 and 3: Divide by total and multiply by 100
    percentages = {key: np.round((value / total) * 100, 2) for key, value in values_dict.items()}

    
    return percentages