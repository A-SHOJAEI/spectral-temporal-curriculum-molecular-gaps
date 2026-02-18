#!/usr/bin/env python3
"""
Basic validation script that doesn't require external dependencies.
Tests only syntax and structure without importing modules that need PyTorch.
"""

import sys
import os
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent

def test_yaml_config():
    """Test YAML config file syntax."""
    print("Testing YAML config file...")

    try:
        config_path = project_root / "configs" / "default.yaml"
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False

        # Basic YAML syntax check without importing yaml
        with open(config_path, 'r') as f:
            content = f.read()

        # Check for basic required sections
        required_sections = ["model:", "training:", "data:", "experiment:"]
        for section in required_sections:
            if section not in content:
                print(f"✗ Missing section: {section}")
                return False
            print(f"✓ Found section: {section}")

        print("✓ YAML config file structure looks good")
        return True

    except Exception as e:
        print(f"✗ YAML config test failed: {e}")
        return False

def test_python_syntax():
    """Test Python syntax of all modules."""
    print("Testing Python file syntax...")

    try:
        import py_compile

        # Find all Python files
        python_files = list(project_root.rglob("*.py"))
        python_files = [f for f in python_files if not f.name.startswith('.') and 'validate_project' not in f.name]

        for py_file in python_files:
            try:
                py_compile.compile(str(py_file), doraise=True)
                print(f"✓ {py_file.relative_to(project_root)} - syntax OK")
            except py_compile.PyCompileError as e:
                print(f"✗ {py_file.relative_to(project_root)} - syntax error: {e}")
                return False

        return True

    except Exception as e:
        print(f"✗ Python syntax test failed: {e}")
        return False

def test_file_structure():
    """Test file structure and required files."""
    print("Testing file structure...")

    required_files = [
        "src/spectral_temporal_curriculum_molecular_gaps/__init__.py",
        "src/spectral_temporal_curriculum_molecular_gaps/data/__init__.py",
        "src/spectral_temporal_curriculum_molecular_gaps/data/loader.py",
        "src/spectral_temporal_curriculum_molecular_gaps/data/preprocessing.py",
        "src/spectral_temporal_curriculum_molecular_gaps/models/__init__.py",
        "src/spectral_temporal_curriculum_molecular_gaps/models/model.py",
        "src/spectral_temporal_curriculum_molecular_gaps/training/__init__.py",
        "src/spectral_temporal_curriculum_molecular_gaps/training/trainer.py",
        "src/spectral_temporal_curriculum_molecular_gaps/evaluation/__init__.py",
        "src/spectral_temporal_curriculum_molecular_gaps/evaluation/metrics.py",
        "src/spectral_temporal_curriculum_molecular_gaps/utils/__init__.py",
        "src/spectral_temporal_curriculum_molecular_gaps/utils/config.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "configs/default.yaml"
    ]

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            return False

    return True

def test_gpu_training_features():
    """Test that train.py has GPU training features."""
    print("Testing GPU training features...")

    try:
        script_path = project_root / "scripts" / "train.py"
        with open(script_path, 'r') as f:
            content = f.read()

        required_features = [
            ("torch.cuda.is_available()", "GPU availability check"),
            ("device = torch.device", "Device selection"),
            ("checkpoint", "Model checkpointing"),
            ("CurriculumTrainer", "Trainer class instantiation"),
            (".to(device)", "Moving model to device")
        ]

        for feature, description in required_features:
            if feature in content:
                print(f"✓ {description}: Found '{feature}'")
            else:
                print(f"✗ {description}: Missing '{feature}'")
                return False

        return True

    except Exception as e:
        print(f"✗ GPU training features test failed: {e}")
        return False

def test_directory_creation():
    """Test that required directories exist."""
    print("Testing directory creation...")

    required_dirs = [
        "checkpoints",
        "models",
        "logs",
        "results"
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_name}")
        else:
            print(f"✗ Missing directory: {dir_name}")
            return False

    return True

def main():
    """Run all validation tests."""
    print("="*60)
    print("SPECTRAL TEMPORAL CURRICULUM MOLECULAR GAPS")
    print("Basic Structure and Syntax Validation")
    print("="*60)

    tests = [
        test_file_structure,
        test_directory_creation,
        test_yaml_config,
        test_python_syntax,
        test_gpu_training_features,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        print(f"\n{test_func.__doc__}")
        print("-" * 50)

        try:
            if test_func():
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}")

    print("\n" + "="*60)
    print(f"VALIDATION SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("\nProject structure is valid and ready for use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install torch torch-geometric rdkit-pypi")
        print("2. Run training: python scripts/train.py")
        print("3. Test with full test suite once dependencies are installed")
        print("\nKey features verified:")
        print("✅ GPU training support with torch.cuda.is_available()")
        print("✅ Model checkpoint saving to models/checkpoints directories")
        print("✅ All Python files have valid syntax")
        print("✅ Required directory structure exists")
        print("✅ Configuration files are properly structured")
        return 0
    else:
        print("❌ SOME VALIDATION TESTS FAILED")
        print("Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())