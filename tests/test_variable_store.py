from unittest import main, TestCase

import time
import sympy
import numpy as np

from pygom.model import common_models
from pygom.model.ode_utils import VariableStore, ParameterStore, StateStore
from pygom.model._model_errors import InputError

class AssignTestFloat(float):
    '''A class to ensure that only types we want can be assigned'''
    pass
class TestVariableStore(TestCase):

    def setUp(self):
        pass

    def test_adding_variable(self):
        # create a store
        vs = VariableStore()
        b = sympy.Symbol('b')
        c = sympy.Symbol('c')
        d = sympy.Symbol('d')

        # add a variables one as a string, a named symbol, a raw symbol, as dict
        vs.append('a')
        vs.append('b', symbol=b)
        vs.append(c)
        vs['d'] = d


        # The zeroth variable symbol should be 'a', first b, third c, forth d
        self.assertEqual(str(vs.index[0].symbol), 'a')
        self.assertEqual(vs.index[1].symbol, b)
        self.assertEqual(vs.index[2].symbol, c)
        self.assertEqual(vs.index[3].symbol, d)

        # c should be called c
        self.assertEqual(str(vs.index[2].symbol), 'c')

        # But none should have a value
        self.assertIsNone(vs.index[0].value)
        self.assertIsNone(vs.index[1].value)
        self.assertIsNone(vs.index[2].value)
        self.assertIsNone(vs.index[3].value)

        # and none of the values should be set
        self.assertFalse(vs.all_values_set)

        # We should be able to call by name as well as position
        self.assertEqual(vs.index[0].symbol, vs['a'].symbol)
        self.assertEqual(vs.index[1].symbol, vs['b'].symbol)
        self.assertEqual(vs.index[2].symbol, vs['c'].symbol)
        self.assertEqual(vs.index[3].symbol, vs['d'].symbol)

        # Test updating a variable
        vs['b'] = c
        self.assertEqual(vs.index[2].symbol, vs['b'].symbol)
        self.assertEqual(vs.get_index('b'), 1)

    def test_adding_list(self):
        # create the store
        vs = VariableStore()
        # create some symbols
        b = sympy.Symbol('b')
        c = sympy.Symbol('c')
        d = sympy.Symbol('d')

        # Create a mixed list
        variables = ['a', b, c, d]

        # add the list to the store
        vs.extend(variables)

        # The zeroth variable symbol should be 'a', first b, third c, forth d
        self.assertEqual(str(vs.index[0].symbol), 'a')
        self.assertEqual(vs.index[1].symbol, b)
        self.assertEqual(vs.index[2].symbol, c)
        self.assertEqual(vs.index[3].symbol, d)

        # c should be called c
        self.assertEqual(str(vs.index[2].symbol), 'c')

        # But none should have a value
        self.assertIsNone(vs.index[0].value)
        self.assertIsNone(vs.index[1].value)
        self.assertIsNone(vs.index[2].value)
        self.assertIsNone(vs.index[3].value)

        # None of the values should be set
        self.assertFalse(vs.all_values_set)


    def test_assigning_values_singles(self):
        # create the store
        vs = VariableStore()

        # Create a list of variables
        variables = ['a', 'b', 'c', 'd']

        # Create a list of values
        values = [2, 3, 5, 7]

        # add the variables to the store
        vs.extend(variables)

        # None of the values should be set
        self.assertFalse(vs.all_values_set)

        # assign the variables (in reverse order)
        for variable, value in reversed(list(zip(variables, values))):
            vs.set_value(variable=variable, value=value)

        # The values should be set
        self.assertTrue(vs.all_values_set)

        # Test that the values match up
        for variable, value in zip(variables, values):
            self.assertEqual(vs[variable].value, value)

    def test_assigning_values_list(self):
        # create the store
        vs = VariableStore()

        # Create a list of variables
        variables = ['a', 'b', 'c', 'd']

        # Create a list of values
        values = [2, 3, 5, 7]

        # add the variables to the store
        vs.extend(variables)

        # None of the values should be set
        self.assertFalse(vs.all_values_set)

        # assign the values 
        vs.set_value_list(values)

        # The values should be set
        self.assertTrue(vs.all_values_set)

        # Test that the values match up
        for variable, value in zip(variables, values):
            self.assertEqual(vs[variable].value, value)

    def test_solution_speed(self):
        # create the store
        vs = VariableStore()

        # how big a system to test on
        N = int(1E5) # one hundred thousand parameters / states!
        ALLOWABLE_VAR_CREATION_TIME = 10 # give it 10 seconds to create vars
        ALLOWABLE_VALUE_UPDATE = 1 # 1 second to update the values of vars

        start_time = time.perf_counter()
        # add the variables
        vs.extend(['var_' + n for n in map(str, range(N))])
        value_time = time.perf_counter()
        vs.set_value_list(list(range(N)))
        end_time = time.perf_counter()

        self.assertLess(value_time - start_time, ALLOWABLE_VAR_CREATION_TIME)
        self.assertLess(end_time - value_time, ALLOWABLE_VALUE_UPDATE)

    def test_value_type(self):
        # create the store
        vs = VariableStore(acceptable_value_types=[AssignTestFloat])

        # Create a list of variables
        variables = ['a', 'b', 'c', 'd']

        # add the variables to the store
        vs.extend(variables)

        # Some values that will work and some that will not
        values_error = {'a':2, 'b':3, 'c':5, 'd':7}
        values_correct = {'a':AssignTestFloat(2), 
                          'b':AssignTestFloat(3),
                          'c':AssignTestFloat(5), 
                          'd':AssignTestFloat(7)}

        # add the values to the store (this should fail)
        with self.assertRaises(InputError) as context:
            vs.values=values_error

        vs.values = values_correct

        # Check the assignment
        for variable, value in values_correct.items():
            self.assertEqual(vs[variable].value, value)

    def test_numpy_numeric_values(self):
        numeric_types = [np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64,
                         np.float16, np.float32, np.float64, np.longdouble]
        for store_type in [VariableStore, ParameterStore, StateStore]:
            for numeric_type in numeric_types:
                with self.subTest(store=store_type, dtype=numeric_type):
                    vs = store_type()
                    vs.append('a')
                    value = numeric_type(2)
                    vs.set_value('a', value)
                    self.assertIs(vs['a'].value, value)
                    self.assertTrue(vs.all_values_set)
                    # Updating between NumPy and Python types must keep the
                    # per-type bookkeeping consistent.
                    vs.set_value('a', 3.5)
                    vs.set_value('a', value)
                    vs.set_value('a', None)
                    self.assertFalse(vs.all_values_set)

    def test_numpy_array_values(self):
        for store_type in [VariableStore, ParameterStore, StateStore]:
            for dtype in [np.int64, np.float32]:
                with self.subTest(store=store_type, dtype=dtype):
                    vs = store_type()
                    vs.extend(['a', 'b'])
                    vs.values = np.array([2, 3], dtype=dtype)
                    self.assertEqual(vs.values, [2, 3])
                    self.assertTrue(vs.all_values_set)

    def test_numpy_values_respect_allowed_types(self):
        for value in [np.int64(2), np.float32(2)]:
            with self.subTest(value=value):
                vs = VariableStore(acceptable_value_types=[AssignTestFloat])
                vs.append('a')
                with self.assertRaises(InputError):
                    vs.set_value('a', value)
        # Explicitly accepted NumPy types must continue to be supported.
        vs = VariableStore(acceptable_value_types=[np.int64])
        vs.append('a')
        value = np.int64(2)
        vs.set_value('a', value)
        self.assertIs(vs['a'].value, value)

    def test_numpy_non_real_values_rejected(self):
        for value in [np.bool_(True), np.complex128(2), np.array(2),
                      np.timedelta64(1, 'D'), np.timedelta64('NaT'),
                      np.datetime64('2026-01-01'), np.datetime64('NaT')]:
            with self.subTest(value=value):
                vs = VariableStore()
                vs.append('a')
                with self.assertRaises(InputError):
                    vs.set_value('a', value)

    def test_rejected_value_preserves_bookkeeping(self):
        invalid_values = [np.timedelta64(1, 'D'), np.timedelta64('NaT'),
                          np.datetime64('2026-01-01'), np.datetime64('NaT')]
        for store_type in [VariableStore, ParameterStore, StateStore]:
            for invalid_value in invalid_values:
                with self.subTest(store=store_type, value=invalid_value):
                    vs = store_type()
                    vs.append('a')
                    for value, bucket in [(None, 'NoneType'),
                                          (np.int64(2), 'int'),
                                          (np.float32(3.5), 'float')]:
                        vs.set_value('a', value)
                        with self.assertRaises(InputError):
                            vs.set_value('a', invalid_value)
                        self.assertIs(vs['a'].value, value)
                        self.assertEqual(
                            [key for key, values in vs._values_by_type.items()
                             if 'a' in values], [bucket])
                        self.assertEqual(vs.all_values_set, value is not None)
                    vs.set_value('a', None)
                    self.assertFalse(vs.all_values_set)

    def test_explicit_numpy_timedelta_type(self):
        vs = VariableStore(acceptable_value_types=[np.timedelta64])
        vs.append('a')
        for value in [np.timedelta64(1, 'D'), np.timedelta64('NaT')]:
            with self.subTest(value=value):
                vs.set_value('a', value)
                self.assertIs(vs['a'].value, value)
                self.assertTrue(vs.all_values_set)
                vs.set_value('a', None)
                self.assertFalse(vs.all_values_set)

    def test_model_numpy_time_parameter_rejected(self):
        for value in [np.timedelta64(1, 'D'), np.timedelta64('NaT'),
                      np.datetime64('2026-01-01'), np.datetime64('NaT')]:
            with self.subTest(value=value):
                model = common_models.SIR_norm()
                with self.assertRaises(InputError):
                    model.parameters = {'beta': value, 'gamma': 0.1}
                self.assertIsNone(model.parameters)
                model.parameters = {'beta': np.float32(0.3), 'gamma': 0.1}
                expected = model.parameters
                with self.assertRaises(InputError):
                    model.parameters = {'beta': value}
                self.assertEqual(model.parameters, expected)



# TODO: Test error conditions
# repeat adding of variable
# mismatched values list length
# non-str keys

    def tearDown(self):
        pass


if __name__ == '__main__':
    main()
