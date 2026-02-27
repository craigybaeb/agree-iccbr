import load.nets

def load_net(dataset_name):
    if dataset_name == "adult":
        import load.nets.net_adult as net
    elif dataset_name == "bank":
        import load.nets.net_bank as net
    elif dataset_name == "beans":
        import load.nets.net_beans as net
    elif dataset_name == "cancer":
        import load.nets.net_cancer as net
    elif dataset_name == "heloc":
        import load.nets.net_heloc as net
    elif dataset_name == "mushroom":
        import load.nets.net_mushroom as net
    elif dataset_name == "ocean":
        import load.nets.net_ocean as net
    elif dataset_name == "wine":
        import load.nets.net_wine as net
    else:
        raise ValueError("There is no dataset with this name.")
    
    return net
    