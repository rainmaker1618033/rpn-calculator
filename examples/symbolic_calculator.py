# ============================================================================
# FILE: symbolic_calculator.py
# ============================================================================
"""
Symbolic Calculator built on top of RPN Calculator Library
Requires: sympy library (pip install sympy)
Uses: rpn_calculator package (modular version)
"""

from rpn_calculator import Calculator
import sympy as sp
from typing import Dict, Any, Optional

class SymbolicCalculator:
    """Symbolic calculator using RPN calculator as numerical engine"""
    
    def __init__(self):
        # Disable logging for symbolic calculator's internal RPN use
        self.rpn = Calculator(enable_logging=False)
        self.variables: Dict[str, sp.Symbol] = {}
        self.expressions: Dict[str, sp.Expr] = {}
        
    def define_variable(self, name: str, value=None):
        """Define a symbolic variable"""
        self.variables[name] = sp.Symbol(name)
        if value is not None:
            self.expressions[name] = sp.sympify(value)
        print(f"Defined: {name}" + (f" = {value}" if value else ""))
    
    def parse_expression(self, expr_str: str) -> sp.Expr:
        """Parse string to symbolic expression"""
        return sp.sympify(expr_str, locals=self.variables)
    
    def differentiate(self, expr_str: str, var: str):
        """Symbolic differentiation"""
        expr = self.parse_expression(expr_str)
        if var not in self.variables:
            self.define_variable(var)
        result = sp.diff(expr, self.variables[var])
        print(f"d/d{var}[{expr}] = {result}")
        return result
    
    def integrate(self, expr_str: str, var: str):
        """Symbolic integration"""
        expr = self.parse_expression(expr_str)
        if var not in self.variables:
            self.define_variable(var)
        result = sp.integrate(expr, self.variables[var])
        print(f"∫{expr} d{var} = {result}")
        return result
    
    def simplify(self, expr_str: str):
        """Simplify expression"""
        expr = self.parse_expression(expr_str)
        result = sp.simplify(expr)
        print(f"Simplified: {expr} = {result}")
        return result
    
    def expand(self, expr_str: str):
        """Expand expression"""
        expr = self.parse_expression(expr_str)
        result = sp.expand(expr)
        print(f"Expanded: {expr} = {result}")
        return result
    
    def factor(self, expr_str: str):
        """Factor expression"""
        expr = self.parse_expression(expr_str)
        result = sp.factor(expr)
        print(f"Factored: {expr} = {result}")
        return result
    
    def evaluate_at(self, expr_str: str, **substitutions):
        """Evaluate symbolic expression at specific values using RPN"""
        expr = self.parse_expression(expr_str)
        
        # Substitute symbolic values
        for var_name, value in substitutions.items():
            if var_name in self.variables:
                expr = expr.subs(self.variables[var_name], value)
        
        # Convert to float and evaluate using RPN if needed
        try:
            numeric_value = float(expr)
            print(f"Result: {numeric_value}")
            return numeric_value
        except:
            print(f"Cannot evaluate to number: {expr}")
            return expr
    
    def compute_rpn(self, rpn_expression: str, clear_after=False):
        """Directly use RPN calculator with helper methods"""
        if clear_after:
            result = self.rpn.evaluate_and_clear(rpn_expression)
        else:
            result = self.rpn.evaluate(rpn_expression)
        print(f"RPN Result: {result}")
        return result
    
    def solve_equation(self, equation_str: str, var: str):
        """Solve equation symbolically"""
        if var not in self.variables:
            self.define_variable(var)
        
        # Parse equation (assumes format: "expr1 = expr2" or just "expr")
        if "=" in equation_str:
            lhs, rhs = equation_str.split("=")
            eq = sp.Eq(self.parse_expression(lhs.strip()), 
                      self.parse_expression(rhs.strip()))
        else:
            eq = self.parse_expression(equation_str)
        
        solutions = sp.solve(eq, self.variables[var])
        print(f"Solutions for {var}: {solutions}")
        return solutions
    
    def taylor_series(self, expr_str: str, var: str, point=0, order=5):
        """Compute Taylor series expansion"""
        expr = self.parse_expression(expr_str)
        if var not in self.variables:
            self.define_variable(var)
        
        series = sp.series(expr, self.variables[var], point, order)
        print(f"Taylor series of {expr} around {var}={point}:")
        print(f"  {series}")
        return series
    
    def limit(self, expr_str: str, var: str, point):
        """Compute limit"""
        expr = self.parse_expression(expr_str)
        if var not in self.variables:
            self.define_variable(var)
        
        result = sp.limit(expr, self.variables[var], point)
        print(f"lim({var}→{point}) {expr} = {result}")
        return result
    
    def get_rpn_stack(self):
        """Get current RPN calculator stack"""
        return self.rpn.get_result() if self.rpn.stack else None
    
    def clear_rpn_stack(self):
        """Clear RPN calculator stack"""
        self.rpn.stack.clear()
        print("RPN stack cleared")


# Example demonstrations
def demo_basic():
    """Basic symbolic operations"""
    print("\n" + "="*60)
    print("DEMO: Basic Symbolic Operations")
    print("="*60 + "\n")
    
    calc = SymbolicCalculator()
    
    # Define variables
    print(">>> Defining variables x and y")
    calc.define_variable('x')
    calc.define_variable('y')
    
    print("\n>>> Differentiation:")
    calc.differentiate('x**2 + 3*x + 5', 'x')
    calc.differentiate('sin(x) * cos(x)', 'x')
    
    print("\n>>> Integration:")
    calc.integrate('x**2', 'x')
    calc.integrate('exp(x)', 'x')
    
    print("\n>>> Simplification:")
    calc.simplify('(x**2 - 1)/(x - 1)')
    calc.simplify('sin(x)**2 + cos(x)**2')
    
    print("\n>>> Expansion:")
    calc.expand('(x + y)**3')
    
    print("\n>>> Factoring:")
    calc.factor('x**2 - 4')

def demo_numeric_evaluation():
    """Combine symbolic and numeric"""
    print("\n" + "="*60)
    print("DEMO: Symbolic + Numeric Evaluation")
    print("="*60 + "\n")
    
    calc = SymbolicCalculator()
    calc.define_variable('x')
    
    # Get derivative symbolically
    print(">>> Find derivative of f(x) = x³ - 2x² + x")
    derivative = calc.differentiate('x**3 - 2*x**2 + x', 'x')
    
    # Evaluate at specific points
    print("\n>>> Evaluate derivative at x=2:")
    result = calc.evaluate_at(str(derivative), x=2)
    
    print("\n>>> Evaluate derivative at x=5:")
    result = calc.evaluate_at(str(derivative), x=5)

def demo_equation_solving():
    """Solve equations"""
    print("\n" + "="*60)
    print("DEMO: Equation Solving")
    print("="*60 + "\n")
    
    calc = SymbolicCalculator()
    calc.define_variable('x')
    
    print(">>> Quadratic equation: x² - 5x + 6 = 0")
    calc.solve_equation('x**2 - 5*x + 6 = 0', 'x')
    
    print("\n>>> Cubic equation: x³ - 6x² + 11x - 6 = 0")
    calc.solve_equation('x**3 - 6*x**2 + 11*x - 6 = 0', 'x')
    
    print("\n>>> Transcendental equation: e^x = 10")
    calc.solve_equation('exp(x) = 10', 'x')

def demo_rpn_integration():
    """Use RPN calculator directly with helper methods"""
    print("\n" + "="*60)
    print("DEMO: RPN Calculator Integration (Using Helper Methods)")
    print("="*60 + "\n")
    
    calc = SymbolicCalculator()
    
    print(">>> Complex number magnitude: |3+4i|")
    calc.compute_rpn('3 4 CMPLX ABS', clear_after=True)
    
    print("\n>>> Vector dot product: [1,2,3] · [4,5,6]")
    calc.compute_rpn('[1,2,3] [4,5,6] DOT', clear_after=True)
    
    print("\n>>> Trigonometry: sin(30°)")
    calc.rpn.state.degrees = True
    calc.compute_rpn('30 SIN', clear_after=True)
    
    print("\n>>> Chain calculation: (5 + 3) × 2")
    calc.compute_rpn('5 3 + 2 *', clear_after=True)

def demo_taylor_series():
    """Taylor series expansion"""
    print("\n" + "="*60)
    print("DEMO: Taylor Series Expansions")
    print("="*60 + "\n")
    
    calc = SymbolicCalculator()
    calc.define_variable('x')
    
    print(">>> Taylor series of sin(x) around x=0:")
    calc.taylor_series('sin(x)', 'x', 0, 6)
    
    print("\n>>> Taylor series of e^x around x=0:")
    calc.taylor_series('exp(x)', 'x', 0, 5)
    
    print("\n>>> Taylor series of ln(1+x) around x=0:")
    calc.taylor_series('log(1+x)', 'x', 0, 5)

def demo_limits():
    """Limit calculations"""
    print("\n" + "="*60)
    print("DEMO: Limits")
    print("="*60 + "\n")
    
    calc = SymbolicCalculator()
    calc.define_variable('x')
    
    print(">>> lim(x→0) sin(x)/x")
    calc.limit('sin(x)/x', 'x', 0)
    
    print("\n>>> lim(x→∞) (1 + 1/x)^x")
    calc.limit('(1 + 1/x)**x', 'x', sp.oo)
    
    print("\n>>> lim(x→0) (e^x - 1)/x")
    calc.limit('(exp(x) - 1)/x', 'x', 0)

def demo_combined_workflow():
    """Advanced: Combine symbolic and numeric operations"""
    print("\n" + "="*60)
    print("DEMO: Combined Symbolic-Numeric Workflow")
    print("="*60 + "\n")
    
    calc = SymbolicCalculator()
    calc.define_variable('x')
    
    print(">>> Problem: Find critical points of f(x) = x³ - 6x² + 9x + 1")
    print(">>> Step 1: Find derivative")
    derivative = calc.differentiate('x**3 - 6*x**2 + 9*x + 1', 'x')
    
    print("\n>>> Step 2: Solve f'(x) = 0")
    critical_points = calc.solve_equation(str(derivative) + ' = 0', 'x')
    
    print("\n>>> Step 3: Evaluate f(x) at critical points using RPN")
    for point in critical_points:
        print(f"\nAt x = {point}:")
        # Use RPN to compute x³ - 6x² + 9x + 1
        x_val = float(point)
        calc.rpn.push(x_val)
        calc.rpn.push(3)
        calc.rpn.operations["^"]()
        
        calc.rpn.push(x_val)
        calc.rpn.push(2)
        calc.rpn.operations["^"]()
        calc.rpn.push(6)
        calc.rpn.operations["*"]()
        calc.rpn.operations["-"]()
        
        calc.rpn.push(x_val)
        calc.rpn.push(9)
        calc.rpn.operations["*"]()
        calc.rpn.operations["+"]()
        
        calc.rpn.push(1)
        calc.rpn.operations["+"]()
        
        result = calc.rpn.get_result()
        print(f"  f({point}) = {result}")
        calc.clear_rpn_stack()


def show_menu():
    """Display main menu"""
    print("\n" + "="*60)
    print("SYMBOLIC CALCULATOR")
    print("Built on RPN Calculator Library")
    print("="*60)
    print("\nMENU:")
    print("  1) Differentiate expression")
    print("  2) Integrate expression")
    print("  3) Simplify/Expand/Factor expression")
    print("  4) Solve equation")
    print("  5) Compute limit")
    print("  6) Taylor series expansion")
    print("  7) Evaluate expression at point")
    print("  8) RPN calculator mode")
    print("  9) Show RPN stack")
    print("  D) Run all demos")
    print("  H) Help/Examples")
    print("  Q) Quit")
    print("="*60)


def interactive_mode():
    """Main interactive loop"""
    calc = SymbolicCalculator()
    
    while True:
        show_menu()
        choice = input("\nChoice: ").strip().upper()
        
        if choice == 'Q':
            print("\nGoodbye!")
            break
            
        elif choice == 'D':
            print("\n>>> Running all demonstrations...")
            demo_basic()
            demo_numeric_evaluation()
            demo_equation_solving()
            demo_rpn_integration()
            demo_taylor_series()
            demo_limits()
            demo_combined_workflow()
            input("\nPress Enter to continue...")
            
        elif choice == 'H':
            print("\n" + "="*60)
            print("EXAMPLES AND HELP")
            print("="*60)
            print("\nAvailable Demos:")
            print("  • Basic Operations (differentiation, integration, simplification)")
            print("  • Numeric Evaluation (symbolic → numeric)")
            print("  • Equation Solving (algebraic and transcendental)")
            print("  • RPN Integration (using helper methods)")
            print("  • Taylor Series Expansions")
            print("  • Limits")
            print("  • Combined Workflows")
            print("\nTo see all examples, select 'D' from the main menu.")
            print("\nSymPy Expression Syntax:")
            print("  • Powers: x**2")
            print("  • Functions: sin(x), cos(x), exp(x), log(x)")
            print("  • Constants: pi, E")
            print("  • Operators: +, -, *, /, **")
            input("\nPress Enter to continue...")
            
        elif choice == '1':
            expr = input("Enter expression: ").strip()
            var = input("Differentiate with respect to: ").strip()
            if var not in calc.variables:
                calc.define_variable(var)
            try:
                calc.differentiate(expr, var)
            except Exception as e:
                print(f"Error: {e}")
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            expr = input("Enter expression: ").strip()
            var = input("Integrate with respect to: ").strip()
            if var not in calc.variables:
                calc.define_variable(var)
            try:
                calc.integrate(expr, var)
            except Exception as e:
                print(f"Error: {e}")
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            print("\nOptions: (S)implify, (E)xpand, (F)actor")
            op = input("Choose: ").strip().upper()
            expr = input("Enter expression: ").strip()
            try:
                if op == 'S':
                    calc.simplify(expr)
                elif op == 'E':
                    calc.expand(expr)
                elif op == 'F':
                    calc.factor(expr)
                else:
                    print("Invalid option")
            except Exception as e:
                print(f"Error: {e}")
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            equation = input("Enter equation (e.g., x**2 - 5*x + 6 = 0): ").strip()
            var = input("Solve for variable: ").strip()
            if var not in calc.variables:
                calc.define_variable(var)
            try:
                calc.solve_equation(equation, var)
            except Exception as e:
                print(f"Error: {e}")
            input("\nPress Enter to continue...")
            
        elif choice == '5':
            expr = input("Enter expression: ").strip()
            var = input("Variable: ").strip()
            point = input("Limit as variable approaches (use 'oo' for infinity): ").strip()
            if var not in calc.variables:
                calc.define_variable(var)
            try:
                point_val = sp.oo if point.lower() == 'oo' else sp.sympify(point)
                calc.limit(expr, var, point_val)
            except Exception as e:
                print(f"Error: {e}")
            input("\nPress Enter to continue...")
            
        elif choice == '6':
            expr = input("Enter expression: ").strip()
            var = input("Variable: ").strip()
            point = input("Expand around point (default 0): ").strip()
            order = input("Order (default 5): ").strip()
            if var not in calc.variables:
                calc.define_variable(var)
            try:
                point_val = float(point) if point else 0
                order_val = int(order) if order else 5
                calc.taylor_series(expr, var, point_val, order_val)
            except Exception as e:
                print(f"Error: {e}")
            input("\nPress Enter to continue...")
            
        elif choice == '7':
            expr = input("Enter expression: ").strip()
            print("Enter variable values (e.g., x=2, y=3):")
            values_str = input("Values: ").strip()
            try:
                # Parse values like "x=2, y=3"
                subs = {}
                for pair in values_str.split(','):
                    var, val = pair.split('=')
                    var = var.strip()
                    if var not in calc.variables:
                        calc.define_variable(var)
                    subs[var] = float(val.strip())
                calc.evaluate_at(expr, **subs)
            except Exception as e:
                print(f"Error: {e}")
            input("\nPress Enter to continue...")
            
        elif choice == '8':
            print("\nRPN Calculator Mode (type 'back' to return)")
            print("Examples: '3 4 +', '[1,2,3] [4,5,6] DOT', '30 SIN'")
            while True:
                rpn_expr = input("RPN> ").strip()
                if rpn_expr.lower() == 'back':
                    break
                if rpn_expr:
                    try:
                        calc.compute_rpn(rpn_expr, clear_after=False)
                    except Exception as e:
                        print(f"Error: {e}")
            
        elif choice == '9':
            result = calc.get_rpn_stack()
            if result is not None:
                print(f"\nTop of RPN stack: {result}")
            else:
                print("\nRPN stack is empty")
            print("Full stack:")
            calc.rpn.print_stack()
            input("\nPress Enter to continue...")
            
        else:
            print("Invalid choice. Please try again.")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    print("\nSymbolic Calculator with RPN Integration")
    print("Requires: sympy (pip install sympy)")
    print("Uses: rpn_hp_calc_14g.py\n")
    
    interactive_mode()