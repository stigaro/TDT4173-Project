def list_label_to_string_label(label: list):
    return {
        [1, 0, 0, 0, 0]: 'EXTREMELY NEGATIVE',
        [0, 1, 0, 0, 0]: 'NEGATIVE',
        [0, 0, 1, 0, 0]: 'NEUTRAL',
        [0, 0, 0, 1, 0]: 'POSITIVE',
        [0, 0, 0, 0, 1]: 'EXTREMELY POSITIVE',
    }[label]


def string_label_to_list_label(label: str):
    return {
        'EXTREMELY NEGATIVE': [1, 0, 0, 0, 0],
        'NEGATIVE': [0, 1, 0, 0, 0],
        'NEUTRAL': [0, 0, 1, 0, 0],
        'POSITIVE': [0, 0, 0, 1, 0],
        'EXTREMELY POSITIVE': [0, 0, 0, 0, 1],
    }[label.upper()]


def get_max_length_from_list_of_string(string_list: list):
    return max([len(string) for string in string_list])
