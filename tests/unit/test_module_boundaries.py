"""Tests for module boundary integrity."""

import sys
import importlib.util
from pathlib import Path


class TestModuleBoundaries:
    """Verify core module does not depend on web or streaming modules."""

    def test_core_not_importing_web(self):
        """Verify core module does not import from web."""
        core_modules = [
            'core.models',
            'core.types',
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
            'core.types',
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
            '__future__', 'dataclasses', 'enum', 'typing', 'datetime', 'numpy',
            'config', 'core'
        }

        core_modules = [
            'core.models',
            'core.types',
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

    def test_active_web_modules_do_not_import_legacy_detection_modules(self):
        forbidden = ("ripple", "activity", "heatmap", "spatiotemporal")
        active_modules = [
            Path("src/web/app.py"),
            Path("src/web/detection_worker.py"),
            Path("src/web/schemas.py"),
        ]

        for path in active_modules:
            content = path.read_text(encoding="utf-8").lower()
            for term in forbidden:
                assert term not in content, f"{path} contains legacy detection term: {term}"
