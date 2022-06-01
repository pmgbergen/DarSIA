from daria import hello


def test_helloworld_no_params():
    assert hello() == "Hello world"


def test_helloworld_param():
    assert hello("Erlend") == "Hello Erlend!"
