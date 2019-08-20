from unittest import TestCase

import pygom

class TestPackageBasics(TestCase):

    def test_version(self):
        '''
        Test __version__ exists and does not error
        '''
        self.assertGreater(len(pygom.__version__), 0,
                           '__version__ should not be empty')

