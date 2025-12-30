"""
Download real microscopy datasets for VantaScope.
Handles multiple sources: Google Drive, GitHub, direct URLs.
"""

import requests
import gdown
import subprocess
import h5py
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vantascope.utils.helpers import load_yaml, ensure_dir
from vantascope.utils.logging import logger


class DatasetDownloader:
    """Download and prepare real microscopy datasets."""
    
    def __init__(self, config_path: str = "config/datasets.yaml"):
        self.config = load_yaml(config_path)
        self.data_dir = ensure_dir("data")
    
    def download_all(self) -> None:
        """Download all configured datasets."""
        logger.info("🌐 Downloading VantaScope datasets...")
        
        for dataset_name, dataset_config in self.config['datasets'].items():
            try:
                self.download_dataset(dataset_name, dataset_config)
            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {e}")
                logger.info(f"Continuing with other datasets...")
    
    def download_dataset(self, name: str, config: Dict[str, Any]) -> None:
        """Download a single dataset."""
        data_path = Path(config['data_path'])
        
        if data_path.exists():
            logger.info(f"✅ {name} already exists at {data_path}")
            return
        
        logger.info(f"📥 Downloading {name}...")
        
        download_config = config.get('download', {})
        method = download_config.get('method', 'skip')
        
        if method == "gdown":
            self._download_gdown_single(name, config, download_config)
        elif method == "gdown_multi":
            self._download_gdown_multi(name, config, download_config)
        elif method == "git_clone":
            self._download_git_clone(name, config, download_config)
        else:
            logger.warning(f"Unknown download method for {name}: {method}")
    
    def _download_gdown_single(self, name: str, config: Dict, download_config: Dict) -> None:
        """Download single file from Google Drive."""
        drive_id = download_config['google_drive_id']
        output_path = Path(config['data_path'])
        
        # Download to temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            gdown.download(id=drive_id, output=tmp_path)
            
            # Process if needed
            processing = download_config.get('processing', 'none')
            if processing == "hdf5_to_numpy_timeseries":
                self._convert_hdf5_to_numpy_timeseries(tmp_path, output_path)
            else:
                # Simple copy
                shutil.move(tmp_path, output_path)
            
            logger.info(f"✅ Downloaded {name} to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
    
    def _download_gdown_multi(self, name: str, config: Dict, download_config: Dict) -> None:
        """Download multiple files and combine."""
        files = download_config['files']
        temp_files = {}
        
        try:
            # Download all files
            for file_key, drive_id in files.items():
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{file_key}.npy') as tmp_file:
                    temp_files[file_key] = tmp_file.name
                
                gdown.download(id=drive_id, output=temp_files[file_key])
                logger.info(f"  Downloaded {file_key}")
            
            # Combine files
            processing = download_config.get('processing', 'combine_numpy_to_hdf5')
            if processing == "combine_numpy_to_hdf5":
                self._combine_numpy_to_hdf5(temp_files, Path(config['data_path']))
            
            logger.info(f"✅ Combined {name} dataset")
            
        except Exception as e:
            logger.error(f"Failed to download multi-file {name}: {e}")
        finally:
            # Cleanup temp files
            for temp_path in temp_files.values():
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
    
    def _download_git_clone(self, name: str, config: Dict, download_config: Dict) -> None:
        """Download from git repository."""
        repo_url = download_config['repository']
        file_path = download_config['file_path']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone repository
            logger.info(f"  Cloning {repo_url}...")
            subprocess.run(['git', 'clone', repo_url, temp_dir], check=True, capture_output=True)
            
            # Copy specific file
            source_file = Path(temp_dir) / file_path
            if source_file.exists():
                shutil.copy(source_file, config['data_path'])
                logger.info(f"✅ Downloaded {name} from git")
            else:
                logger.error(f"File not found in repo: {file_path}")
    
    def _convert_hdf5_to_numpy_timeseries(self, hdf5_path: str, output_path: Path) -> None:
        """Convert HDF5 time series to numpy array."""
        logger.info("  Converting HDF5 to numpy time series...")
        
        with h5py.File(hdf5_path, 'r') as f:
            # Find image datasets (heuristic)
            datasets = []
            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2:
                    datasets.append((name, obj))
            f.visititems(collect_datasets)
            
            if datasets:
                # Take every 5th frame to create time series
                frames = []
                for i in range(0, min(50, len(datasets)), 5):  # Max 50 frames, every 5th
                    _, dataset = datasets[i]
                    frame = np.array(dataset)
                    if frame.ndim == 2:
                        frames.append(frame)
                
                if frames:
                    time_series = np.stack(frames, axis=0)
                    np.save(output_path, time_series)
                    logger.info(f"    Created time series: {time_series.shape}")
                else:
                    logger.error("No suitable frames found for time series")
    
    def _combine_numpy_to_hdf5(self, temp_files: Dict[str, str], output_path: Path) -> None:
        """Combine multiple numpy files into single HDF5."""
        logger.info("  Combining numpy files to HDF5...")
        
        with h5py.File(output_path, 'w') as f:
            for key, file_path in temp_files.items():
                data = np.load(file_path)
                f.create_dataset(key, data=data)
                f[key].attrs['source'] = 'gdown_download'
                logger.info(f"    Added {key}: {data.shape}")
    
    def verify_downloads(self) -> None:
        """Verify all datasets were downloaded correctly."""
        logger.info("🔍 Verifying downloaded datasets...")
        
        for dataset_name, dataset_config in self.config['datasets'].items():
            data_path = Path(dataset_config['data_path'])
            
            if data_path.exists():
                file_size = data_path.stat().st_size / (1024**2)  # MB
                logger.info(f"✅ {dataset_name}: {file_size:.1f} MB")
            else:
                logger.warning(f"❌ {dataset_name}: Not found")


def main():
    """Download all datasets."""
    downloader = DatasetDownloader()
    
    try:
        downloader.download_all()
        downloader.verify_downloads()
        logger.info("🎉 Dataset download complete!")
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Download failed: {e}")


if __name__ == "__main__":
    main()
