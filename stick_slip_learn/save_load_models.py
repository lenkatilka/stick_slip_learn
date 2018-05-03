

def save_model(model, filename, *args, **kwargs):

    import _pickle
    with open(filename, 'wb') as f:
        _pickle.dump(model, f)

def load_model(filename, *args, **kwargs):
    import _pickle

    with open(filename, 'rb') as f:
        _pickle.load(model, f)

    return model
