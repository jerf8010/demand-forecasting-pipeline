from datetime import datetime
def preprocess_text():
    """ Logg Preprocess steps

    How to use
    ----------
    print(f'{preprocess_text()} Messenger... ')

    """
    return f'> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} Process:'
