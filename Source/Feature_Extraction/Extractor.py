def integer_label_to_string_label(label: int):
    return {
        -2: 'EXTREMELY NEGATIVE',
        -1: 'NEGATIVE',
        0: 'NEUTRAL',
        1: 'POSITIVE',
        2: 'EXTREMELY POSITIVE',
    }[label]


def string_label_to_integer_label(label: str):
    return {
        'EXTREMELY NEGATIVE': -2,
        'NEGATIVE': -1,
        'NEUTRAL': 0,
        'POSITIVE': 1,
        'EXTREMELY POSITIVE': 2,
    }[label.upper()]


def get_max_length_from_list_of_string(string_list: list):
    return max([len(string) for string in string_list])
