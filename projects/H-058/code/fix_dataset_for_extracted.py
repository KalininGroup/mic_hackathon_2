import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Let's create a simple fix by modifying the _index_files method
def create_fixed_dataset():
    """Create a version that works with extracted directories."""
    
    # Read the current file
    with open('src/vantascope/data/dft_graphene_loader.py', 'r') as f:
        content = f.read()
    
    # Replace the _index_files method
    old_method = '''    def _index_files(self):
        """Index H5 files by split (train/test folders)."""
        self.h5_files = []
        
        logger.info("📦 Reading from BigGrapheneDataset.tar.gz...")
        with tarfile.open(self.data_path, 'r:gz') as tar:
            all_files = tar.getnames()
            
            # Filter by split folder
            self.h5_files = [
                f for f in all_files 
                if f.endswith('.h5') and f.startswith(f'{self.split}/')
            ]'''
    
    new_method = '''    def _index_files(self):
        """Index H5 files by split (train/test folders)."""
        self.h5_files = []
        
        # Check if data_path is a .tar.gz file or extracted directory
        if str(self.data_path).endswith('.tar.gz'):
            logger.info("📦 Reading from BigGrapheneDataset.tar.gz...")
            with tarfile.open(self.data_path, 'r:gz') as tar:
                all_files = tar.getnames()
                
                # Filter by split folder
                self.h5_files = [
                    f for f in all_files 
                    if f.endswith('.h5') and f.startswith(f'{self.split}/')
                ]
        else:
            # Handle extracted directory
            from pathlib import Path
            logger.info(f"📁 Reading from extracted directory: {self.data_path}")
            
            if self.split == 'train':
                split_dir = Path(self.data_path).parent / 'train'
            else:
                split_dir = Path(self.data_path).parent / 'test'
                
            self.h5_files = [str(f) for f in split_dir.glob('*.h5')]'''
    
    # Replace in content
    new_content = content.replace(old_method, new_method)
    
    # Write back
    with open('src/vantascope/data/dft_graphene_loader.py', 'w') as f:
        f.write(new_content)
    
    print("✅ Dataset fixed to handle extracted directories!")

if __name__ == "__main__":
    create_fixed_dataset()
