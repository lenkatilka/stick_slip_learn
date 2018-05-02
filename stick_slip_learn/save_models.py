

def save_model(model, filename, *args, **kwargs):

    import _pickle
    with open(filename, 'wb') as f:
        _pickle.dump(model, f)
