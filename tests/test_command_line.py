from unittest import TestCase

import stick_slip_learn
from stick_slip_learn.command_line import main

class test_console(TestCase):
    def test_basic(self):
        main()
