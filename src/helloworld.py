def hello(name=None) -> str:
    if name is None:
        return "Hello world"
    else:
        return f"Hello {name}!"
