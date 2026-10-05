//! Standalone entry point for native protocol validation.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    jesse_coordinator::run()
}
