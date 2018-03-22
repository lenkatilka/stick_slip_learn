from unittest import TestCase

import stick_slip_learn
from stick_slip_learn.command_line import main

class test_string(TestCase):
    def is_string(self):
        s = stick_slip_learn.text()
        self.assertTrue(isinstance(s,basestring))
        
class test_console(TestCase):
    def test_basic(self):
        main()
