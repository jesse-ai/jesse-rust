import importlib, sys

try:
    rust_mod = importlib.import_module("jesse_rust")
    # bring symbols into package namespace
    globals().update({name: getattr(rust_mod, name) for name in dir(rust_mod) if not name.startswith("__")})
except ImportError:
    # If the compiled module is not available, provide a fallback or error message
    print("Warning: Rust native module 'jesse_rust' not compiled. Run 'maturin develop' in jesse-rust directory.")
    # You could also raise an exception or provide Python fallbacks here
    pass