import unittest

from typing import Callable, Any


class TestCase(unittest.TestCase):
    def assertRaisesChain(self, exceptions: list[type], f: Callable,
            *nargs, **kwargs):
        cm: Any
        with self.assertRaises(exceptions[0]) as cm:
            f(*nargs, **kwargs)

        current = cm.exception
        for e in exceptions:
            self.assertIsInstance(current, e)
            current = current.__cause__
