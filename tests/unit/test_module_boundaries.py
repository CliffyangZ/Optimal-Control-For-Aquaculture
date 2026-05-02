"""Tests for module boundary integrity."""

import sys
import importlib.util


class TestModuleBoundaries:
    """Verify core module does not depend on web or streaming modules."""

    def test_core_not_importing_web(self):
        """Verify core module does not import from web."""
        core_modules = [
            'core.models',
            'core.activity_levels'
        ]

        for module_name in core_modules:
            module = importlib.import_module(module_name)
            module_source = importlib.util.find_spec(module_name).origin

            with open(module_source, 'r') as f:
                content = f.read()

            assert 'from web' not in content, f"{module_name} imports from web"
            assert 'import web' not in content, f"{module_name} imports web"

    def test_core_not_importing_streaming(self):
        """Verify core module does not import from streaming."""
        core_modules = [
            'core.models',
            'core.activity_levels'
        ]

        for module_name in core_modules:
            module = importlib.import_module(module_name)
            module_source = importlib.util.find_spec(module_name).origin

            with open(module_source, 'r') as f:
                content = f.read()

            assert 'from streaming' not in content, f"{module_name} imports from streaming"
            assert 'import streaming' not in content, f"{module_name} imports streaming"

    def test_core_dependencies_minimal(self):
        """Verify core depends only on standard library and approved packages."""
        allowed_imports = {
            'dataclasses', 'enum', 'typing', 'datetime', 'numpy',
            'config', 'core'
        }

        core_modules = [
            'core.models',
            'core.activity_levels'
        ]

        for module_name in core_modules:
            module = importlib.import_module(module_name)
            module_source = importlib.util.find_spec(module_name).origin

            with open(module_source, 'r') as f:
                lines = f.readlines()

            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                if stripped.startswith(('import ', 'from ')):
                    parts = stripped.split()
                    if len(parts) > 1:
                        first_word = parts[1].split('.')[0]
                        if first_word and first_word not in allowed_imports:
                            raise AssertionError(f"{module_name} imports forbidden module: {first_word}")
