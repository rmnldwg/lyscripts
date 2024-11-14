"""Define loadable dummy model for testing."""

class DummyModel:
    def __init__(self, was_externally_loaded: bool = False):
        self.was_externally_loaded = was_externally_loaded

model = DummyModel(was_externally_loaded=True)
