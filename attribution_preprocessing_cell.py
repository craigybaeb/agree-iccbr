# Attribution Preprocessing Cell - Add this BEFORE the case align calculation cell

print("Preprocessing attribution files to expected format...")

# Function to process complex attribution files
def process_attribution_file(attr_path, expected_samples, expected_features):
    """
    Process attribution file and convert to expected 2D format (samples, features).
    Handles various input formats including 4D arrays from complex IG outputs.
    """
    attr = np.load(attr_path)
    print(f"    Raw shape: {attr.shape}")
    
    # Handle different input formats
    if len(attr.shape) == 2:
        # Already in correct format (samples, features)
        if attr.shape == (expected_samples, expected_features):
            print(f"    ✓ Already correct format")
            return attr
        else:
            print(f"    ⚠️ Shape mismatch: {attr.shape} != {(expected_samples, expected_features)}")
            return None
    
    elif len(attr.shape) == 4:
        # Complex format: (samples, steps, features, ?) 
        # Take mean over integration steps and last dimension
        print(f"    Processing 4D format...")
        if attr.shape[0] == expected_samples and attr.shape[2] == expected_features:
            # Average over steps (dim 1) and last dimension (dim 3)
            processed = np.mean(attr, axis=(1, 3))  # Shape: (samples, features)
            print(f"    ✓ Processed to: {processed.shape}")
            return processed
        else:
            print(f"    ❌ Cannot match expected dimensions")
            return None
    
    elif len(attr.shape) == 3:
        # 3D format: (samples, steps, features) or (samples, features, methods)
        print(f"    Processing 3D format...")
        if attr.shape[0] == expected_samples:
            if attr.shape[2] == expected_features:
                # (samples, steps, features) - average over steps
                processed = np.mean(attr, axis=1)
            elif attr.shape[1] == expected_features:
                # (samples, features, methods) - average over methods  
                processed = np.mean(attr, axis=2)
            else:
                print(f"    ❌ Cannot match feature dimension")
                return None
            print(f"    ✓ Processed to: {processed.shape}")
            return processed
    
    else:
        print(f"    ❌ Unsupported shape: {attr.shape}")
        return None

# Test the preprocessing function
test_path = explanation_base_dir / "adult_model1" / "test" / "attributions.npy"
if test_path.exists():
    processed_attr = process_attribution_file(test_path, 1000, 12)
    if processed_attr is not None:
        print(f"\n✅ Attribution preprocessing successful!")
        print(f"   Original file: 4D complex format")
        print(f"   Processed: {processed_attr.shape} - ready for case align")
        print(f"   Range: [{processed_attr.min():.4f}, {processed_attr.max():.4f}]")
        print(f"   Mean: {processed_attr.mean():.4f}")
    else:
        print(f"\n❌ Could not process attribution file")
else:
    print(f"\n❌ Test attribution file not found: {test_path}")