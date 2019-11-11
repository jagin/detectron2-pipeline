from pipeline.pipeline import Pipeline


class AllNumbers(Pipeline):
    """Generate integer numbers"""

    def generator(self):
        value = 0
        while True:
            yield value
            value += 1


class Evens(Pipeline):
    """Filter even numbers only"""

    def filter(self, value):
        return value % 2 == 0


class MultipleOf(Pipeline):
    """Filter numbers which are multiplied by given factor"""

    def __init__(self, factor=1):
        self.factor = factor
        super(MultipleOf, self).__init__()

    def filter(self, value):
        return value % self.factor == 0


class First(Pipeline):
    """Get first 'total' numbers"""

    def __init__(self, total=10):
        self.total = total
        self.count = 0
        super(First, self).__init__()

    def map(self, value):
        self.count += 1
        return value

    def has_next(self):
        return self.count < self.total


class Printer(Pipeline):
    """Print result"""

    def map(self, value):
        print(value)
        return value


class TestPipeline:
    def test_pipeline(self, capsys):
        # Create pipeline modules
        all_numbers = AllNumbers()
        evens = Evens()
        multiple_of_3 = MultipleOf(3)
        printer = Printer()
        first_10 = First(10)

        # Create pipeline
        pipeline = all_numbers | evens | multiple_of_3 | first_10 | printer

        # Iterate through pipeline
        for _ in pipeline:
            pass
        captured = capsys.readouterr()

        assert captured.out == "0\n6\n12\n18\n24\n30\n36\n42\n48\n54\n"
