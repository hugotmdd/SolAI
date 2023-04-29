import collections


def flatten_dict(dictionary, parent_key="", separator="_"):
    """
    Recursively flattens a nested dictionary into a single level dictionary
    with flattened keys that contain the original nested key names separated by the separator.

    Args:
        dictionary (dict): The nested dictionary to be flattened.
        parent_key (str): The key name of the parent dictionary if exists. Defaults to "".
        separator (str): The separator to be used between nested key names. Defaults to "_".

    Returns:
        dict: The flattened dictionary.
    """
    flattened_items = []
    for key, value in dictionary.items():
        # Construct new key name by appending the current key to the parent key, separated by the separator
        new_key = parent_key + separator + key if parent_key else key

        if isinstance(value, collections.MutableMapping):
            # If the current value is a nested dictionary, recursively flatten it
            flattened_items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            # If the current value is not a dictionary, append it as a key-value pair to the flattened_items list
            flattened_items.append((new_key, value))

    # Convert flattened_items list to dictionary and return
    return dict(flattened_items)


# Source: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys