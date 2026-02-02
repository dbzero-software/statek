# pylint: disable=no-member,too-few-public-methods,unused-argument,unused-variable
from statek.task import find_locals
from statek.system import inject_context


class TestFindLocals:
    """Tests for find_locals function."""

    def test_find_all_locals(self):
        """Test finding all local variables when no filters are specified."""
        x = 10
        y = 'hello'
        z = [1, 2, 3]

        results = list(find_locals())

        # Should find at least x, y, z (may include self from test method)
        assert x in results
        assert y in results
        assert z in results

    def test_find_by_type_int(self):
        """Test finding variables by type (int)."""
        x = 10
        y = 20
        z = 'hello'

        results = list(find_locals(var_type=int))

        assert x in results
        assert y in results
        assert z not in results

    def test_find_by_type_str(self):
        """Test finding variables by type (str)."""
        x = 10
        y = 'hello'
        z = 'world'

        results = list(find_locals(var_type=str))

        assert x not in results
        assert y in results
        assert z in results

    def test_find_by_type_list(self):
        """Test finding variables by type (list)."""
        x = [1, 2, 3]
        y = [4, 5]
        z = 'not a list'

        results = list(find_locals(var_type=list))

        assert x in results
        assert y in results
        assert z not in results

    def test_find_by_name(self):
        """Test finding variables by name."""
        user = {'name': 'Alice'}
        admin = {'name': 'Bob'}

        results = list(find_locals(var_name='user'))

        assert len(results) == 1
        assert results[0] == user

    def test_find_by_name_not_found(self):
        """Test finding variables by name when it doesn't exist."""
        x = 10
        y = 20

        results = list(find_locals(var_name='nonexistent'))

        assert len(results) == 0

    def test_find_by_type_and_name(self):
        """Test finding variables by both type and name."""
        user = 42
        admin = 'Alice'

        results = list(find_locals(var_type=int, var_name='user'))

        assert len(results) == 1
        assert results[0] == 42

    def test_find_by_type_and_name_no_match(self):
        """Test finding variables by type and name when type doesn't match."""
        user = 'Alice'
        admin = 42

        results = list(find_locals(var_type=int, var_name='user'))

        assert len(results) == 0

    def test_find_custom_class(self):
        """Test finding variables by custom class type."""
        class User:
            def __init__(self, name):
                self.name = name

        user1 = User('Alice')
        user2 = User('Bob')
        admin = 'not a user'

        results = list(find_locals(var_type=User))

        assert user1 in results
        assert user2 in results
        assert admin not in results
        assert len(results) == 2

    def test_find_with_none_values(self):
        """Test finding variables that have None value."""
        x = None
        y = 10

        results = list(find_locals(var_type=type(None)))

        assert x in results
        assert y not in results

    def test_find_empty_context(self):
        """Test finding variables when only self exists (in test method)."""
        # Only 'self' and 'results' should exist
        results = list(find_locals(var_name='nonexistent'))

        assert len(results) == 0

    def test_find_with_local_context(self):
        """Test finding variables with __local_context extending the scope."""
        # Create additional context
        context_data = {
            'context_var': 42,
            'message': 'from context'
        }

        def function_with_context(**kwargs):
            _local_context = kwargs.get('_local_context')
            x = 10
            y = 'local'
            # Should find both local vars and context vars
            return list(find_locals(var_type=int))

        # Use inject_context wrapper to inject context_data
        f = inject_context(function_with_context, context_data)
        results = f()

        assert 10 in results  # x from local scope
        assert 42 in results  # context_var from __local_context
