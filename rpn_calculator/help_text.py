# ============================================================================
# FILE: rpn_calculator/help_text.py (PAGINATED VERSION)
# ============================================================================
"""Help text and documentation - Paginated version"""

import shutil

# Organize help into categories and subcategories
HELP_SECTIONS = {
    "Overview": """
╔═══════════════════════════════════════════════════════════════════════════╗
║              RPN CALCULATOR - QUICK REFERENCE                             ║
╚═══════════════════════════════════════════════════════════════════════════╝

Type 'HELP <category>' for detailed help on:

BASICS:
  General      - Stack operations, basic commands (C, DEL, UNDO, SWAP, etc.)
  Math         - Arithmetic, scientific functions (+, -, *, /, ^, MOD, etc.)
  Constants    - Mathematical and physical constants (E, PI, I, etc.)

ADVANCED MATH:
  Trig         - Trigonometric functions (SIN, COS, TAN, ASIN, etc.)
  Complex      - Complex number operations (CMPLX, RECT, POLAR, etc.)
  Stats        - Statistics (COMB, PERM, STDV)
  
VECTORS & MATRICES:
  Vector       - Vector operations (DOT, VMAG, VCROSS, VNORM)
  Matrix       - Matrix operations (MATRIX, DET, TRACE, MINV, M+, M*, etc.)
  Decomp       - Matrix decompositions (LU, QR, SVD, CHOLESKY, etc.)
  
SIGNAL PROCESSING:
  FFT          - Fourier transforms (FFT, IFFT, FFT_MAG, FFT_PHASE)
  Signal       - Convolution and correlation (CONV, CONV2, DECONV, XCORR)

SETTINGS:
  Modes        - DEG/RAD mode, display settings
  Examples     - Usage examples

Type 'HELP ALL' to see all help at once (long output!)
Type 'HELP SEARCH <term>' to search for a command
Type '@' to exit the calculator
""",

    "General": """
═══════════════════════════════════════════════════════════════════════════
GENERAL COMMANDS
═══════════════════════════════════════════════════════════════════════════
Stack Operations:
  C            Clear entire stack
  DEL          Delete top item from stack
  UNDO         Undo last operation (restore previous stack state)
  (blank)      Press Enter on blank line to duplicate top of stack
  
Stack Manipulation:
  R↓  (RD)     Roll stack down (bottom item moves to top)
  R↑  (RU)     Roll stack up (top item moves to bottom)
  X<>Y (SWAP)  Swap top two stack items
  
Program Control:
  @            Exit calculator
  H, HELP      Show help menu
  
Examples:
  3 4 SWAP     → Stack: [4, 3]
  1 2 3 RD     → Stack: [3, 1, 2]
  5            → (Enter on empty line duplicates 5)
""",

    "Math": """
═══════════════════════════════════════════════════════════════════════════
MATH & SCIENTIFIC FUNCTIONS
═══════════════════════════════════════════════════════════════════════════
Basic Arithmetic:
  +  -  *  /   Standard operations
  ^            Power (exponentiation)
  MOD          Modulo
  ||           Parallel operation: (x*y)/(x+y)
  
Logarithmic & Exponential:
  LOG          Base-10 logarithm
  LOG2         Base-2 logarithm
  LN           Natural logarithm
  EXP          e^x
  SQRT         Square root
  1/X  (INV)   Reciprocal
  
Integer Operations:
  GCD          Greatest common divisor
  LCM          Least common multiple
  FRAC         Convert to fraction with residual
  
Examples:
  3 4 +        → 7
  2 8 ^        → 256
  12 18 GCD    → 6
  0.75 FRAC    → 3/4
""",

    "Trig": """
═══════════════════════════════════════════════════════════════════════════
TRIGONOMETRIC FUNCTIONS
═══════════════════════════════════════════════════════════════════════════
Forward Functions:
  SIN  COS  TAN       Standard trig functions
  
Inverse Functions:
  ASIN ACOS ATAN      Arc/inverse trig functions
  
Mode Settings:
  DEG                 Set to degrees mode
  RAD                 Set to radians mode
  SHOWMODE            Display current mode
  
Notes:
  • Input angles in current mode (DEG or RAD)
  • Inverse functions return angles in current mode
  • Works with complex numbers for extended domain
  
Examples:
  DEG 30 SIN          → 0.5
  RAD 1.5708 COS      → 0 (π/2)
  0.5 ASIN            → 30 (in DEG mode)
""",

    "Complex": """
═══════════════════════════════════════════════════════════════════════════
COMPLEX NUMBER OPERATIONS
═══════════════════════════════════════════════════════════════════════════
Creating Complex Numbers:
  CMPLX         Combine real and imaginary: real imag CMPLX → a+bj
  RECT          Polar to rectangular: r θ RECT → a+bj
  
Extracting Components:
  RE            Real part
  IM            Imaginary part
  ABS           Magnitude (absolute value)
  ARG           Angle/argument/phase
  
Conversions:
  POLAR         Rectangular to polar: a+bj POLAR → r, θ
  CONJ          Complex conjugate
  
Examples:
  3 4 CMPLX     → 3+4j
  3+4j ABS      → 5
  1 45 RECT     → 0.707+0.707j (in DEG mode)
  3+4j POLAR    → 5, 53.13 (in DEG mode)
""",

    "Vector": """
═══════════════════════════════════════════════════════════════════════════
VECTOR OPERATIONS
═══════════════════════════════════════════════════════════════════════════
Creating Vectors:
  [1,2,3]       Enter vector using square brackets
  
Vector Operations:
  DOT           Dot product (inner product)
  VMAG          Vector magnitude (Euclidean norm)
  VCROSS        Cross product (3D vectors only)
  VNORM         Normalize vector (unit vector)
  
Element-wise Operations:
  + - * / ^     Work element-wise on vectors
                Scalars broadcast automatically
  
Examples:
  [1,2,3] [4,5,6] DOT    → 32
  [3,4] VMAG             → 5
  [1,0,0] [0,1,0] VCROSS → [0,0,1]
  [3,4] VNORM            → [0.6, 0.8]
  [1,2,3] 2 *            → [2,4,6]
""",

    "Matrix": """
═══════════════════════════════════════════════════════════════════════════
MATRIX OPERATIONS
═══════════════════════════════════════════════════════════════════════════
Creating Matrices:
  MATRIX        Create from rows: [r1] [r2] ... n MATRIX
  IDENTITY      Create n×n identity matrix: n IDENTITY
  
Properties:
  MSIZE         Get dimensions → rows, cols
  TRANSPOSE     Transpose (rows ↔ columns)
  DET           Determinant
  TRACE         Trace (sum of diagonal)
  RANK          Matrix rank
  COND          Condition number (stability measure)
  NORM          Matrix norm (Frobenius by default)
  
Arithmetic:
  M+            Matrix addition
  M-            Matrix subtraction
  M*            Matrix multiplication (or scalar × matrix)
  MSCALE        Scale by scalar: matrix scalar MSCALE
  
Advanced:
  MINV          Matrix inverse
  EIGEN         Eigenvalues
  EIGENVEC      Eigenvalues and eigenvectors
  RREF          Reduced row echelon form
  MSOLVE        Solve Ax=b: A b MSOLVE → x
  
Examples:
  [1,2] [3,4] 2 MATRIX    → Create 2×2 matrix
  [[1,2],[3,4]] DET       → -2
  [[1,2],[3,4]] MINV      → Inverse matrix
  [[2,1],[1,3]] [5,6] MSOLVE → [1.8, 1.4]
""",

    "Decomp": """
═══════════════════════════════════════════════════════════════════════════
MATRIX DECOMPOSITIONS
═══════════════════════════════════════════════════════════════════════════
Standard Decompositions:
  LU            LU decomposition: A = P·L·U
                Returns P (permutation), L (lower), U (upper)
  
  QR            QR decomposition: A = Q·R
                Q is orthogonal, R is upper triangular
                
  SVD           Singular Value Decomposition: A = U·Σ·V*
                Returns U, Σ (diagonal), V* (conjugate transpose)
  
Specialized:
  CHOLESKY      Cholesky: A = L·L* (positive definite matrices)
                Returns L (lower triangular)
  
  SCHUR         Schur: A = Q·T·Q* (requires scipy)
                T is upper triangular
  
  HESSENBERG    Hessenberg: A = Q·H·Q* (requires scipy)
                H is upper Hessenberg form
  
Usage Notes:
  • All decompositions verify properties before computation
  • Results pushed to stack in order shown
  • Use M* to verify: multiply factors to recover original
  • Some decompositions require scipy library
  
Examples:
  [[1,2],[3,4]] QR        → Returns Q, R
  [[4,2],[2,3]] CHOLESKY  → Returns L
  [[3,2],[2,3]] SVD       → Returns U, Σ, V*
""",

    "FFT": """
═══════════════════════════════════════════════════════════════════════════
FOURIER TRANSFORM OPERATIONS
═══════════════════════════════════════════════════════════════════════════
Transform Operations:
  FFT           Fast Fourier Transform
                Converts time/spatial domain → frequency domain
                Auto zero-pads to next power of 2
  
  IFFT          Inverse FFT
                Converts frequency domain → time/spatial domain
                Returns real values when appropriate
  
Spectral Analysis:
  FFT_MAG       Magnitude spectrum
                Returns |FFT(x)| for each frequency bin
  
  FFT_PHASE     Phase spectrum
                Returns angle of FFT(x)
                Respects DEG/RAD mode
  
Key Features:
  • Automatic zero-padding to power of 2 (reports if done)
  • Conjugate symmetry for real inputs
  • Round-trip: FFT → IFFT recovers original signal
  
FFT Output Order:
  [DC, freq1, freq2, ..., Nyquist, -freq2, -freq1]
  
Examples:
  [1,2,3,4,5,6,7,8] FFT      → Frequency domain
  [1,2,3,4,5] FFT            → Auto-pads to 8 samples
  [signal] FFT_MAG           → Magnitude spectrum
  [signal] FFT IFFT          → Roundtrip (recovers signal)
  
  DEG [1,1,0,0] FFT_PHASE    → Phase in degrees
""",

    "Signal": """
═══════════════════════════════════════════════════════════════════════════
SIGNAL PROCESSING OPERATIONS
═══════════════════════════════════════════════════════════════════════════

1D Convolution:
  CONV          Discrete convolution of two vectors
  CONVOLVE      Alias for CONV
  
  Usage: vector1 vector2 CONV
  Result length = len(vector1) + len(vector2) - 1
  
  Applications:
  • Filtering signals (moving average, smoothing)
  • Edge detection
  • System response analysis
  • Polynomial multiplication

2D Convolution:
  CONV2         2D convolution (for matrices/images)
  
  Usage: matrix kernel CONV2
  
  Applications:
  • Image filtering and blurring
  • Edge detection in images
  • Sharpening
  • Pattern recognition
  
  Common kernels:
  • Box blur:      [[1,1,1],[1,1,1],[1,1,1]]
  • Gaussian:      [[1,2,1],[2,4,2],[1,2,1]]
  • Sharpen:       [[0,-1,0],[-1,5,-1],[0,-1,0]]
  • Sobel X:       [[-1,0,1],[-2,0,2],[-1,0,1]]
  • Sobel Y:       [[-1,-2,-1],[0,0,0],[1,2,1]]

Deconvolution:
  DECONV        Inverse of convolution
  
  Usage: convolved_signal original_signal DECONV
  
  If c = a ⊗ b, then DECONV(c, b) ≈ a
  
  Applications:
  • Remove blurring/filtering
  • System identification
  • Channel equalization
  • Polynomial division
  
  Note: Sensitive to noise; works best with exact known signals

Cross-Correlation:
  XCORR         Cross-correlation of two vectors
  
  Usage: vector1 vector2 XCORR
  
  Applications:
  • Pattern matching
  • Time-delay estimation
  • Signal similarity measurement
  • Auto-correlation (signal with itself)

Examples:
  # 1D Convolution
  [1,2,3] [1,1,1] CONV              → [1, 3, 6, 5, 3]
  [1,2,3,4,5] [0.5,0.5] CONV        → Smooth signal
  
  # Verify with deconvolution
  [1,2,3] [1,1,1] CONV              → [1, 3, 6, 5, 3]
  [1,1,1] DECONV                    → [1, 2, 3]
  
  # 2D Convolution (image processing)
  [[1,2,3],[4,5,6],[7,8,9]] [[1,1],[1,1]] CONV2
  
  # Edge detection
  image [[-1,0,1],[-2,0,2],[-1,0,1]] CONV2
  
  # Auto-correlation
  [1,2,3,4] [1,2,3,4] XCORR

Workflow Examples:
  # Apply and remove filter
  signal [0.5,0.5] CONV [0.5,0.5] DECONV
  
  # Polynomial operations
  [1,2,1] [1,3,2] CONV              → Multiply polynomials
  [1,5,10,7,2] [1,2,1] DECONV       → Divide polynomials
""",

    "Stats": """
═══════════════════════════════════════════════════════════════════════════
STATISTICS OPERATIONS
═══════════════════════════════════════════════════════════════════════════
Combinatorics:
  COMB          Combinations: n k COMB → n choose k
  PERM          Permutations: n k PERM → P(n,k)
  
Descriptive Statistics:
  STDV          Standard deviation of vector
  
Examples:
  5 2 COMB      → 10 (ways to choose 2 from 5)
  5 2 PERM      → 20 (ways to arrange 2 from 5)
  [1,2,3,4,5] STDV → 1.414...
""",

    "Constants": """
═══════════════════════════════════════════════════════════════════════════
MATHEMATICAL CONSTANTS
═══════════════════════════════════════════════════════════════════════════
  E             Euler's number (2.71828...)
  PI            Pi (3.14159...)
  I             Imaginary unit (√-1)

Examples:
  E LN          → 1
  PI 2 /        → 1.5708... (π/2)
  0 1 CMPLX     → 0+1j (same as I)
""",

    "Modes": """
═══════════════════════════════════════════════════════════════════════════
CALCULATOR MODES & SETTINGS
═══════════════════════════════════════════════════════════════════════════
Angle Mode:
  DEG           Set trigonometric mode to degrees
  RAD           Set trigonometric mode to radians
  SHOWMODE      Display current mode and settings
  
Display Settings:
  DIGITS n      Set decimal precision (e.g., DIGITS 4)
  FORMAT FLOAT  Use normal floating-point display
  FORMAT SCIENTIFIC  Use scientific notation (e.g., 1.23e-4)
  
Current Settings:
  • Angle mode affects: SIN, COS, TAN, ASIN, ACOS, ATAN
  • Angle mode affects: RECT, POLAR, ARG, FFT_PHASE
  • Display settings affect stack printout only
""",

    "Examples": """
═══════════════════════════════════════════════════════════════════════════
USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════════════
Basic Arithmetic:
  3 4 +                    → 7
  10 3 - 2 *               → 14 (chain operations)
  
Complex Numbers:
  3 4 CMPLX ABS            → 5 (magnitude of 3+4j)
  
Vectors:
  [1,2,3] [4,5,6] DOT      → 32
  [2,3,4] 2 *              → [4,6,8]
  
Matrices:
  [[1,2],[3,4]] DET        → -2
  [[1,2],[3,4]] MINV       → Inverse matrix
  
Solving Systems:
  [2,1] [1,3] 2 MATRIX     → Create coefficient matrix
  [5,6]                    → Right-hand side
  MSOLVE                   → [1.8, 1.4]
  
FFT:
  [1,2,3,4,5,6,7,8] FFT_MAG  → Frequency spectrum
  [signal] FFT IFFT          → Identity (recovers signal)
  
Stack Operations:
  3 (Enter on blank line)    → Duplicate: [3, 3]
  1 2 3 SWAP                 → [1, 3, 2]
""",
}


def get_terminal_height():
    """Get the terminal height, with fallback to default."""
    try:
        return shutil.get_terminal_size().lines
    except:
        return 24  # Default fallback


def show_paged_text(text, lines_per_page=None):
    """
    Display text with pagination if it exceeds screen height.
    
    Args:
        text: The text to display
        lines_per_page: Number of lines per page (None = auto-detect minus 2 for prompt)
    """
    if lines_per_page is None:
        terminal_height = get_terminal_height()
        lines_per_page = max(10, terminal_height - 2)  # Leave room for prompt
    
    lines = text.split('\n')
    total_lines = len(lines)
    
    # If content fits on one screen, just print it
    if total_lines <= lines_per_page:
        print(text)
        return
    
    # Otherwise, page through it
    line_num = 0
    while line_num < total_lines:
        # Calculate this page
        page_end = min(line_num + lines_per_page, total_lines)
        
        # Display this page
        for i in range(line_num, page_end):
            print(lines[i])
        
        line_num = page_end
        
        # If there's more, prompt to continue
        if line_num < total_lines:
            remaining = total_lines - line_num
            try:
                response = input(f"\n[{remaining} more lines] Press Enter to continue, 'q' to quit: ")
                if response.lower().strip() in ['q', 'quit', 'exit']:
                    print("[Help display interrupted]")
                    break
                print()  # Blank line before next page
            except (KeyboardInterrupt, EOFError):
                print("\n[Help display interrupted]")
                break


def show_instructions(choice=None):
    """Display calculator instructions or a specific section with pagination."""
    if choice is None:
        # Show overview/menu - this is short enough to not need pagination
        print(HELP_SECTIONS["Overview"])
        
    elif choice.upper() == "ALL":
        # Show all sections with pagination
        print("\n" + "="*79)
        print("COMPLETE HELP DOCUMENTATION")
        print("="*79 + "\n")
        
        all_text_parts = []
        for section_name, content in HELP_SECTIONS.items():
            if section_name != "Overview":
                all_text_parts.append(content)
                all_text_parts.append("")  # Blank line between sections
        
        all_text = "\n".join(all_text_parts)
        show_paged_text(all_text)
        
    elif choice.upper().startswith("SEARCH"):
        # Search functionality - already handles output well
        search_term = choice[6:].strip().upper()
        if not search_term:
            print("Usage: HELP SEARCH <term>")
            print("Example: HELP SEARCH matrix")
            return
        
        print(f"\nSearching for '{search_term}'...\n")
        found = False
        results = []
        
        for section_name, content in HELP_SECTIONS.items():
            if search_term in content.upper():
                results.append(f"Found in section: {section_name}")
                # Show just the relevant part
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if search_term in line.upper():
                        # Show context (3 lines before and after)
                        start = max(0, i-3)
                        end = min(len(lines), i+4)
                        results.append("  ...")
                        for j in range(start, end):
                            marker = ">>> " if j == i else "    "
                            results.append(marker + lines[j])
                        results.append("  ...")
                        results.append("")
                found = True
        
        if found:
            show_paged_text("\n".join(results))
        else:
            print(f"No help found for '{search_term}'")
            print("Try: HELP (to see available categories)")
            
    else:
        # Show specific section - try case-insensitive match
        key = choice.strip()
        
        # Try exact match first (case-insensitive)
        matched_key = None
        for section_name in HELP_SECTIONS.keys():
            if section_name.upper() == key.upper():
                matched_key = section_name
                break
        
        if matched_key:
            show_paged_text(HELP_SECTIONS[matched_key])
        else:
            print(f"Unknown help topic: {choice}")
            print("\nAvailable topics:")
            for name in HELP_SECTIONS.keys():
                if name != "Overview":
                    print(f"  {name}")
            print("\nType 'HELP' to see the menu")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
if __name__ == "__main__":
    # Test the pagination
    print("Testing pagination...\n")
    
    # Test 1: Short section (no pagination)
    print("=" * 79)
    print("TEST 1: Short section (Overview)")
    print("=" * 79)
    show_instructions("Overview")
    
    input("\n[Press Enter for next test]")
    
    # Test 2: Long section (with pagination)
    print("\n" + "=" * 79)
    print("TEST 2: Long section (Signal)")
    print("=" * 79)
    show_instructions("Signal")
    
    input("\n[Press Enter for next test]")
    
    # Test 3: Search (may be paginated)
    print("\n" + "=" * 79)
    print("TEST 3: Search")
    print("=" * 79)
    show_instructions("SEARCH matrix")