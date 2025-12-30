"""
Fix the _load_h5_file method to work with extracted directories
"""

def create_fixed_loader():
    """Fix the _load_h5_file method."""
    
    # Read the current file
    with open('src/vantascope/data/dft_graphene_loader.py', 'r') as f:
        content = f.read()
    
    # Replace the _load_h5_file method
    old_method = '''    def _load_h5_file(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Load single H5 file with atomic structure."""
        try:
            with tarfile.open(self.data_path, 'r:gz') as tar:
                h5_bytes = tar.extractfile(file_path).read()
                with h5py.File(io.BytesIO(h5_bytes), 'r') as f:
                    return self._extract_structure(f)
                    
        except Exception as e:
            logger.debug(f"Failed to load {file_path}: {e}")
            return None'''
    
    new_method = '''    def _load_h5_file(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Load single H5 file with atomic structure."""
        try:
            # Check if data_path is a .tar.gz file or extracted directory
            if str(self.data_path).endswith('.tar.gz'):
                # Original tar.gz method
                with tarfile.open(self.data_path, 'r:gz') as tar:
                    h5_bytes = tar.extractfile(file_path).read()
                    with h5py.File(io.BytesIO(h5_bytes), 'r') as f:
                        return self._extract_structure(f)
            else:
                # Extracted directory method
                from pathlib import Path
                # file_path is just the filename, need to construct full path
                if self.split == 'train':
                    full_path = Path(self.data_path).parent / 'train' / file_path
                else:
                    full_path = Path(self.data_path).parent / 'test' / file_path
                
                with h5py.File(full_path, 'r') as f:
                    return self._extract_structure(f)
                    
        except Exception as e:
            logger.debug(f"Failed to load {file_path}: {e}")
            return None'''
    
    # Replace in content
    new_content = content.replace(old_method, new_method)
    
    # Write back
    with open('src/vantascope/data/dft_graphene_loader.py', 'w') as f:
        f.write(new_content)
    
    print("✅ _load_h5_file method fixed!")

if __name__ == "__main__":
    create_fixed_loader()
