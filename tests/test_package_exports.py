"""Tests for root package exports."""


def test_shared_context_public_helpers_are_exported_from_package_root():
    """Shared-context guide examples can import public helpers from statek."""
    from statek import (  # pylint: disable=import-outside-toplevel
        ContextCategory,
        ContextCategoryDict,
        ContextVar,
        init_shared_context,
        print_locals,
        shared_context_set_var,
    )
    from statek.shared_context import (  # pylint: disable=import-outside-toplevel
        ContextCategory as ModuleContextCategory,
        ContextCategoryDict as ModuleContextCategoryDict,
        ContextVar as ModuleContextVar,
        init_shared_context as module_init_shared_context,
        print_locals as module_print_locals,
        shared_context_set_var as module_shared_context_set_var,
    )

    assert ContextCategory is ModuleContextCategory
    assert ContextCategoryDict is ModuleContextCategoryDict
    assert ContextVar is ModuleContextVar
    assert init_shared_context is module_init_shared_context
    assert print_locals is module_print_locals
    assert shared_context_set_var is module_shared_context_set_var
