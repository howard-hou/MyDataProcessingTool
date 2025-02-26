
def round_dict_value(d, n=3):
    o = dict()
    for k, v in d.items():
        if isinstance(v, float):
            o[k] = round(v, n)
        else:
            o[k] = v
    return o