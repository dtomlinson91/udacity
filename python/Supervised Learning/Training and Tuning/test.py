import functools
from time import perf_counter


class slowDown(object):
    """docstring for slowDown"""
    def __init__(self, rate):
        if callable(rate):
            self.func = rate
            self.rate = 1
        else:
            self.rate = rate

    def __get__(self, instance, owner=None):
        return functools.partial(self, instance)

    def __call__(self, *args, **kwargs):
        if not hasattr(self, 'func'):
            self.func = args[0]
            return self
        start_time = perf_counter()
        self.func(*args, **kwargs)
        end_time = perf_counter()
        run_time = end_time - start_time
        print(f'Finished {self.func.__name__!r} in {run_time:.4f} secs')
