from collections import OrderedDict


def copy_state_dict(state_dict):
    result = OrderedDict()
    for key, value in state_dict.items():
        result[key] = value.clone().detach().cpu()
    return result
