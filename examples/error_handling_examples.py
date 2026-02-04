#!/usr/bin/env python3
"""
RPN Calculator - Error Handling Examples
Demonstrates proper error handling patterns for robust applications

run as: python error_handling_examples.py
"""
from rpn_calculator import Calculator, CalculatorError


def example_basic_error_handling():
    """Example 1: Basic Error Handling Pattern"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Error Handling")
    print("=" * 70)
    
    calc = Calculator(enable_logging=False)
    
    # Try a division by zero
    try:
        result = calc.evaluate_and_clear("10 0 /")
        print(f"Result: {result}")
    except CalculatorError as e:
        print(f"❌ Error caught: {e.message}")
        print("✓ Application continues running safely")
    
    print()


def example_invalid_operations():
    """Example 2: Handling Invalid Operations"""
    print("=" * 70)
    print("EXAMPLE 2: Invalid Operations")
    print("=" * 70)
    
    calc = Calculator(enable_logging=False)
    
    # Not enough operands
    try:
        result = calc.evaluate_and_clear("5 +")  # Missing second operand
        print(f"Result: {result}")
    except CalculatorError as e:
        print(f"❌ Not enough operands: {e.message}")
    
    # Invalid token
    try:
        result = calc.evaluate_and_clear("5 INVALID_OP")
        print(f"Result: {result}")
    except CalculatorError as e:
        print(f"❌ Invalid operation: {e.message}")
    
    print()


def example_csv_processing_with_errors():
    """Example 3: CSV Processing with Robust Error Handling"""
    print("=" * 70)
    print("EXAMPLE 3: CSV Processing with Error Handling")
    print("=" * 70)
    
    import csv
    import os
    import tempfile
    
    # Create sample CSV with some problematic data
    # Use Windows-compatible temp directory
    temp_dir = tempfile.gettempdir()
    csv_file = os.path.join(temp_dir, 'sample_with_errors.csv')
    
    csv_data = [
        ['x', 'y', 'operation'],
        ['10', '2', 'x y /'],        # Valid: 10/2 = 5
        ['5', '0', 'x y /'],          # Error: division by zero
        ['3', '4', 'x y + 2 /'],      # Valid: (3+4)/2 = 3.5
        ['abc', '5', 'x y +'],        # Error: invalid number
        ['8', '2', 'x y ^ SQRT'],     # Valid: sqrt(8^2) = 8
    ]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
    
    print(f"Processing CSV file: {csv_file}")
    print()
    
    calc = Calculator(enable_logging=False)
    results = []
    error_count = 0
    success_count = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (row 1 is header)
            x_str = row['x']
            y_str = row['y']
            operation = row['operation']
            
            # Replace placeholders with actual values
            expr = operation.replace('x', x_str).replace('y', y_str)
            
            try:
                # Attempt to evaluate the expression
                result = calc.evaluate_and_clear(expr)
                success_count += 1
                
                print(f"✓ Row {row_num}: {x_str}, {y_str} → {operation} = {result:.4f}")
                results.append({
                    'row': row_num,
                    'x': x_str,
                    'y': y_str,
                    'result': result,
                    'status': 'success'
                })
                
            except CalculatorError as e:
                error_count += 1
                print(f"❌ Row {row_num}: {x_str}, {y_str} → ERROR: {e.message}")
                results.append({
                    'row': row_num,
                    'x': x_str,
                    'y': y_str,
                    'result': None,
                    'status': 'error',
                    'error_message': e.message
                })
            
            except ValueError as e:
                # Handle conversion errors (invalid numbers)
                error_count += 1
                print(f"❌ Row {row_num}: {x_str}, {y_str} → VALUE ERROR: Invalid number format")
                results.append({
                    'row': row_num,
                    'x': x_str,
                    'y': y_str,
                    'result': None,
                    'status': 'error',
                    'error_message': f'Invalid number: {str(e)}'
                })
    
    print()
    print(f"Summary: {success_count} successful, {error_count} errors")
    print()
    
    # Clean up
    if os.path.exists(csv_file):
        os.remove(csv_file)


def example_graceful_degradation():
    """Example 4: Graceful Degradation Pattern"""
    print("=" * 70)
    print("EXAMPLE 4: Graceful Degradation")
    print("=" * 70)
    
    calc = Calculator(enable_logging=False)
    
    def safe_calculate(expression, default=None):
        """
        Safely evaluate an expression, returning default value on error.
        This pattern is useful when you want the program to continue
        even if some calculations fail.
        """
        try:
            return calc.evaluate_and_clear(expression)
        except CalculatorError as e:
            print(f"  Warning: {e.message} - using default value: {default}")
            return default
    
    # Process a batch of calculations
    calculations = [
        ("3 4 +", "sum"),
        ("10 0 /", "division by zero"),  # Will error
        ("5 2 ^", "power"),
        ("INVALID", "bad syntax"),        # Will error
        ("2 3 * 4 +", "complex"),
    ]
    
    results = {}
    for expr, name in calculations:
        result = safe_calculate(expr, default=0)
        results[name] = result
        if result is not None:
            print(f"✓ {name:20} = {result}")
    
    print()
    print(f"Processed {len(calculations)} calculations")
    print(f"Valid results: {sum(1 for v in results.values() if v is not None)}")
    print()


def example_retry_pattern():
    """Example 5: Retry Pattern with Fallback"""
    print("=" * 70)
    print("EXAMPLE 5: Retry Pattern with Fallback")
    print("=" * 70)
    
    calc = Calculator(enable_logging=False)
    
    def calculate_with_fallback(primary_expr, fallback_expr):
        """
        Try primary expression first, fall back to alternative if it fails.
        Useful for trying optimal method first, with a simpler backup.
        """
        try:
            result = calc.evaluate_and_clear(primary_expr)
            print(f"✓ Primary succeeded: {primary_expr} = {result}")
            return result
        except CalculatorError as e:
            print(f"  Primary failed ({e.message}), trying fallback...")
            try:
                result = calc.evaluate_and_clear(fallback_expr)
                print(f"✓ Fallback succeeded: {fallback_expr} = {result}")
                return result
            except CalculatorError as e2:
                print(f"❌ Both methods failed: {e2.message}")
                raise
    
    # Example: Try inverse, fall back to alternative calculation
    print("Attempting calculation with fallback:")
    
    # This will work
    result1 = calculate_with_fallback("4 2 /", "4 2 -")
    print()
    
    # This will use fallback (trying to invert zero matrix would fail)
    # For demonstration, we'll use a simpler example
    result2 = calculate_with_fallback("10 0 /", "10 1 /")
    print()


def example_validation_pattern():
    """Example 6: Input Validation Before Calculation"""
    print("=" * 70)
    print("EXAMPLE 6: Input Validation Pattern")
    print("=" * 70)
    
    calc = Calculator(enable_logging=False)
    
    def validate_and_calculate(x, y, operation):
        """
        Validate inputs before attempting calculation.
        This prevents errors and provides better user feedback.
        """
        # Input validation
        errors = []
        
        # Check if numbers are valid
        try:
            x_val = float(x)
        except (ValueError, TypeError):
            errors.append(f"Invalid x value: '{x}'")
        
        try:
            y_val = float(y)
        except (ValueError, TypeError):
            errors.append(f"Invalid y value: '{y}'")
        
        # Check for specific problematic cases
        if operation == '/' and str(y) == '0':
            errors.append("Division by zero not allowed")
        
        # If validation failed, report all errors
        if errors:
            print(f"❌ Validation failed:")
            for error in errors:
                print(f"   - {error}")
            return None
        
        # Validation passed, perform calculation
        try:
            expr = f"{x} {y} {operation}"
            result = calc.evaluate_and_clear(expr)
            print(f"✓ {x} {operation} {y} = {result}")
            return result
        except CalculatorError as e:
            print(f"❌ Calculation error: {e.message}")
            return None
    
    # Test cases
    test_cases = [
        (10, 2, '+'),      # Valid
        (10, 0, '/'),      # Caught by validation
        ('abc', 5, '*'),   # Invalid input
        (8, 2, '^'),       # Valid
    ]
    
    for x, y, op in test_cases:
        print(f"\nTesting: {x} {op} {y}")
        validate_and_calculate(x, y, op)
    
    print()


def example_logging_errors():
    """Example 7: Logging Errors for Debugging"""
    print("=" * 70)
    print("EXAMPLE 7: Error Logging Pattern")
    print("=" * 70)
    
    import logging
    from datetime import datetime
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    calc = Calculator(enable_logging=False)
    
    def calculate_with_logging(expression, context=""):
        """
        Calculate with comprehensive error logging.
        Useful for production systems where you need audit trails.
        """
        logger.info(f"Starting calculation: '{expression}' {context}")
        
        try:
            result = calc.evaluate_and_clear(expression)
            logger.info(f"Success: {expression} = {result}")
            return result
            
        except CalculatorError as e:
            logger.error(f"CalculatorError in '{expression}': {e.message}")
            logger.debug(f"Context: {context}")
            return None
            
        except Exception as e:
            logger.critical(f"Unexpected error in '{expression}': {str(e)}")
            logger.debug(f"Context: {context}", exc_info=True)
            return None
    
    # Run calculations with logging
    calculations = [
        ("3 4 +", "User: Alice, Session: 12345"),
        ("10 0 /", "User: Bob, Session: 67890"),
        ("5 2 ^", "User: Alice, Session: 12345"),
    ]
    
    for expr, context in calculations:
        calculate_with_logging(expr, context)
    
    print()


def example_batch_processing():
    """Example 8: Batch Processing with Error Summary"""
    print("=" * 70)
    print("EXAMPLE 8: Batch Processing with Error Summary")
    print("=" * 70)
    
    calc = Calculator(enable_logging=False)
    
    # Simulated batch of expressions
    expressions = [
        "3 4 +",
        "10 5 /",
        "2 8 ^",
        "15 0 /",      # Error
        "5 INVALID",   # Error
        "7 3 -",
        "9 SQRT",
        "0 1 /",
        "BADTOKEN",    # Error
        "12 3 MOD",
    ]
    
    results = []
    errors = []
    
    print(f"Processing {len(expressions)} expressions...\n")
    
    for i, expr in enumerate(expressions, 1):
        try:
            result = calc.evaluate_and_clear(expr)
            results.append({
                'index': i,
                'expression': expr,
                'result': result,
                'status': 'success'
            })
            print(f"  {i:2}. ✓ {expr:20} = {result}")
            
        except CalculatorError as e:
            errors.append({
                'index': i,
                'expression': expr,
                'error': e.message,
                'status': 'failed'
            })
            print(f"  {i:2}. ❌ {expr:20} → {e.message}")
    
    # Summary report
    print()
    print("=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total expressions:  {len(expressions)}")
    print(f"Successful:         {len(results)} ({len(results)/len(expressions)*100:.1f}%)")
    print(f"Failed:             {len(errors)} ({len(errors)/len(expressions)*100:.1f}%)")
    
    if errors:
        print("\nFailed Expressions:")
        for err in errors:
            print(f"  [{err['index']}] {err['expression']:20} - {err['error']}")
    
    print()


def main():
    """Run all error handling examples"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  RPN CALCULATOR - ERROR HANDLING EXAMPLES".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    try:
        example_basic_error_handling()
        example_invalid_operations()
        example_csv_processing_with_errors()
        example_graceful_degradation()
        example_retry_pattern()
        example_validation_pattern()
        example_logging_errors()
        example_batch_processing()
        
        print("=" * 70)
        print("ALL ERROR HANDLING EXAMPLES COMPLETED")
        print("=" * 70)
        print()
        print("Key Takeaways:")
        print("  • Always use try-except blocks with CalculatorError")
        print("  • Validate inputs before calculation when possible")
        print("  • Provide meaningful error messages to users")
        print("  • Use logging for production systems")
        print("  • Implement graceful degradation for non-critical errors")
        print("  • Consider retry/fallback patterns for robustness")
        print()
        
    except Exception as e:
        print(f"\n❌ Unexpected error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()