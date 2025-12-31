"""
Run all tests

Execute this file to run the complete test suite
"""

import sys
import traceback

def run_all_tests():
    """Run all test modules"""
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE TEST SUITE")
    print("=" * 80 + "\n")
    
    tests = [
        ("Component Tests", "test_components"),
        ("Encoder-Decoder Integration Tests", "test_encoder_decoder"),
        ("SQuAD Workflow Tests", "test_squad_workflow"),
    ]
    
    results = []
    
    for test_name, module_name in tests:
        print(f"\n{'=' * 80}")
        print(f"Running: {test_name}")
        print(f"{'=' * 80}\n")
        
        try:
            if module_name == "test_components":
                from test_components import run_component_tests
                run_component_tests()
            elif module_name == "test_encoder_decoder":
                from test_encoder_decoder import run_tests
                run_tests()
            elif module_name == "test_squad_workflow":
                from test_squad_workflow import test_squad_workflow
                test_squad_workflow()
            
            results.append((test_name, "PASSED", None))
            print(f"\n✓ {test_name} completed successfully")
            
        except Exception as e:
            results.append((test_name, "FAILED", str(e)))
            print(f"\n✗ {test_name} failed with error:")
            traceback.print_exc()
    
    # Print summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    
    for test_name, status, error in results:
        status_icon = "✓" if status == "PASSED" else "✗"
        print(f"{status_icon} {test_name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓✓✓")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print(f"SOME TESTS FAILED ({failed}/{len(results)})")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
