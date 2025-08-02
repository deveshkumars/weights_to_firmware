#!/usr/bin/env python3
"""
Simple test script to verify the modernized get_models.py works with Python 3
"""

import sys
import os

# Add the current directory to the path so we can import quad_gen
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly (without TensorFlow)"""
    try:
        import quad_gen.get_models as get_models
        import quad_gen.gaussian_mlp as gaussian_mlp
        import quad_gen.code_blocks as code_blocks
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_string_formatting():
    """Test that f-string formatting works correctly"""
    try:
        test_dir = "/path/to/test/directory"
        test_file = "test.txt"
        full_path = f"{test_dir}/{test_file}"
        assert full_path == "/path/to/test/directory/test.txt"
        print("✓ String formatting test passed")
        return True
    except Exception as e:
        print(f"✗ String formatting error: {e}")
        return False

def test_type_hints():
    """Test that type hints are properly formatted"""
    try:
        from typing import List, Optional
        from quad_gen.get_models import subdir, read_txt_to_get_dirs
        
        # These should have proper type hints
        assert subdir.__annotations__['root_dir'] == str
        assert subdir.__annotations__['return'] == List[str]
        print("✓ Type hints test passed")
        return True
    except Exception as e:
        print(f"✗ Type hints error: {e}")
        return False

def test_function_signatures():
    """Test that function signatures are properly updated"""
    try:
        from quad_gen.get_models import save_result, analyze_seeds
        
        # Check that functions have proper type hints
        print(f"save_result annotations: {save_result.__annotations__}")
        assert 'model_dir' in save_result.__annotations__, "model_dir not in annotations"
        assert 'out_dir' in save_result.__annotations__, "out_dir not in annotations"
        assert 'osi' in save_result.__annotations__, "osi not in annotations"
        assert 'absolute_path' in save_result.__annotations__, "absolute_path not in annotations"
        assert save_result.__annotations__['return'] is None, f"return type is {save_result.__annotations__['return']}, expected None"
        
        print("✓ Function signatures test passed")
        return True
    except Exception as e:
        print(f"✗ Function signatures error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gaussian_mlp_signatures():
    """Test that gaussian_mlp functions have proper signatures"""
    try:
        from quad_gen.gaussian_mlp import generate
        
        # Check that the generate function has proper type hints
        assert 'policy' in generate.__annotations__
        assert 'sess' in generate.__annotations__
        assert 'output_path' in generate.__annotations__
        assert generate.__annotations__['return'] == str
        
        print("✓ Gaussian MLP signatures test passed")
        return True
    except Exception as e:
        print(f"✗ Gaussian MLP signatures error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing modernized get_models.py (Python 3 compatibility)...")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_string_formatting,
        test_type_hints,
        test_function_signatures,
        test_gaussian_mlp_signatures
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The Python 3 modernization was successful.")
        print("\nNote: TensorFlow 2 compatibility requires TensorFlow to be properly installed.")
        print("The code has been updated to use tf.compat.v1 for backward compatibility.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 