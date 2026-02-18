#!/usr/bin/env python3
"""
Basic validation script to test core functionality without external dependencies.
This script tests basic imports and functionality to ensure the code works.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_basic_imports():
    """Test that basic imports work."""
    print("Testing basic imports...")

    try:
        from spectral_temporal_curriculum_molecular_gaps.utils.config import (
            get_default_config,
            validate_config,
            load_config,
            save_config,
            merge_configs
        )
        print("✓ Utils config imports successful")

        # Test config functionality
        default_config = get_default_config()
        validate_config(default_config)
        print("✓ Default config validation successful")

        # Test config merging
        override_config = {"model": {"hidden_dim": 128}}
        merged = merge_configs(default_config, override_config)
        assert merged["model"]["hidden_dim"] == 128
        print("✓ Config merging works")

    except Exception as e:
        print(f"✗ Utils config test failed: {e}")
        return False

    return True

def test_config_loading():
    """Test config file loading."""
    print("Testing config file loading...")

    try:
        from spectral_temporal_curriculum_molecular_gaps.utils.config import load_config

        # Test loading the default config file
        config_path = project_root / "configs" / "default.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
            print("✓ Default config file loaded successfully")

            # Validate required sections exist
            required_sections = ["model", "training", "data", "experiment"]
            for section in required_sections:
                if section not in config:
                    print(f"✗ Missing required section: {section}")
                    return False
            print("✓ All required config sections present")

            return True
        else:
            print(f"✗ Config file not found: {config_path}")
            return False

    except Exception as e:
        print(f"✗ Config loading test failed: {e}")
        return False

def test_scripts_syntax():
    """Test that main scripts have valid syntax."""
    print("Testing script syntax...")

    try:
        import py_compile

        scripts = [
            project_root / "scripts" / "train.py",
            project_root / "scripts" / "evaluate.py"
        ]

        for script in scripts:
            if script.exists():
                py_compile.compile(str(script), doraise=True)
                print(f"✓ {script.name} syntax is valid")
            else:
                print(f"✗ Script not found: {script}")
                return False

        return True

    except Exception as e:
        print(f"✗ Script syntax test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist."""
    print("Testing directory structure...")

    required_dirs = [
        "src/spectral_temporal_curriculum_molecular_gaps",
        "src/spectral_temporal_curriculum_molecular_gaps/data",
        "src/spectral_temporal_curriculum_molecular_gaps/models",
        "src/spectral_temporal_curriculum_molecular_gaps/training",
        "src/spectral_temporal_curriculum_molecular_gaps/evaluation",
        "src/spectral_temporal_curriculum_molecular_gaps/utils",
        "scripts",
        "tests",
        "configs",
        "checkpoints",
        "models"
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            return False

    return True

def test_train_script_gpu_check():
    """Test that train.py properly checks for GPU availability."""
    print("Testing GPU availability check in train.py...")

    try:
        script_path = project_root / "scripts" / "train.py"
        with open(script_path, 'r') as f:
            content = f.read()

        # Check for GPU-related code
        gpu_checks = [
            "torch.cuda.is_available()",
            "cuda",
            "device",
            "checkpoints"
        ]

        for check in gpu_checks:
            if check in content:
                print(f"✓ Found GPU-related code: {check}")
            else:
                print(f"✗ Missing GPU-related code: {check}")
                return False

        return True

    except Exception as e:
        print(f"✗ GPU check test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("SPECTRAL TEMPORAL CURRICULUM MOLECULAR GAPS")
    print("Basic Functionality Validation")
    print("="*50)

    tests = [
        test_directory_structure,
        test_basic_imports,
        test_config_loading,
        test_scripts_syntax,
        test_train_script_gpu_check,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        print(f"\n{test_func.__doc__}")
        print("-" * 40)

        try:
            if test_func():
                passed += 1
                print("✓ PASSED")
            else:
                print("✗ FAILED")
        except Exception as e:
            print(f"✗ FAILED with exception: {e}")

    print("\n" + "="*50)
    print(f"SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All basic validation tests passed!")
        print("\nNext steps:")
        print("1. Install required dependencies (PyTorch, PyTorch Geometric, etc.)")
        print("2. Run actual training with: python scripts/train.py")
        print("3. Run comprehensive tests with proper test dependencies")
        return 0
    else:
        print("✗ Some validation tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())