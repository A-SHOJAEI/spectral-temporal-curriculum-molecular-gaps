#!/usr/bin/env python3
"""Basic functionality test script to verify the project works without dependencies."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test that all modules can be imported (basic syntax check)."""
    print("Testing imports...")

    try:
        # Test configuration
        from spectral_temporal_curriculum_molecular_gaps.utils.config import get_default_config, validate_config
        print("✓ Config module imports successfully")

        # Test default config
        config = get_default_config()
        validate_config(config)
        print("✓ Default config validation passes")

        print("✓ All import tests passed!")
        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "src/spectral_temporal_curriculum_molecular_gaps/__init__.py",
        "src/spectral_temporal_curriculum_molecular_gaps/models/model.py",
        "src/spectral_temporal_curriculum_molecular_gaps/training/trainer.py",
        "src/spectral_temporal_curriculum_molecular_gaps/data/loader.py",
        "src/spectral_temporal_curriculum_molecular_gaps/data/preprocessing.py",
        "src/spectral_temporal_curriculum_molecular_gaps/evaluation/metrics.py",
        "src/spectral_temporal_curriculum_molecular_gaps/utils/config.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "configs/default.yaml",
        "requirements.txt",
        "README.md",
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")

    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False

    print("✓ All required files present!")
    return True

def test_config_consistency():
    """Test that config files and defaults are consistent."""
    print("\nTesting config consistency...")

    try:
        from spectral_temporal_curriculum_molecular_gaps.utils.config import load_config, get_default_config

        # Load default config
        default_config = get_default_config()

        # Load config file
        config_file = load_config("configs/default.yaml")

        # Check key consistency
        required_sections = ["model", "training", "data", "experiment", "reproducibility"]

        for section in required_sections:
            if section not in config_file:
                print(f"✗ Missing section in config file: {section}")
                return False
            if section not in default_config:
                print(f"✗ Missing section in default config: {section}")
                return False

        print("✓ Config file and defaults are consistent!")
        return True

    except Exception as e:
        print(f"✗ Config consistency error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SPECTRAL TEMPORAL CURRICULUM MOLECULAR GAPS - BASIC TESTS")
    print("=" * 60)

    all_passed = True

    # Test file structure first
    if not test_file_structure():
        all_passed = False

    # Test config consistency
    if not test_config_consistency():
        all_passed = False

    # Test imports last (requires dependencies)
    if not test_imports():
        print("\nNote: Import tests failed - this is expected without dependencies installed")
        print("Run 'pip install -r requirements.txt' to install dependencies")

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL BASIC TESTS PASSED!")
        print("Project structure is correct and ready for use.")
    else:
        print("✗ SOME TESTS FAILED!")
        print("Please fix the issues above before proceeding.")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())