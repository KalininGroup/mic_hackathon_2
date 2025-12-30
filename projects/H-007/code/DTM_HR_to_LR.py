!pip install pyro5
!pip install scifireaders
!pip install sidpy
!pip install pynsid
!pip install git+https://github.com/pycroscopy/DTMicroscope.git
!pip install matplotlib

!run_server_afm

import matplotlib.pylab as plt
import numpy as np
import Pyro5.api
from IPython.display import clear_output, display
import Pyro5.errors
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

!wget https://github.com/pycroscopy/DTMicroscope/raw/boris_dev/DTMicroscope/test/datasets/dset_spm1.h5

uri = "PYRO:microscope.server@localhost:9092"
mic = Pyro5.api.Proxy(uri)
mic._pyroBind()

print("[OK] connected:", mic._pyroUri)
print("\n[REMOTE METHODS]")
for m in sorted(mic._pyroMethods):
    print(" -", m)

print("\n[REMOTE ATTRS]")
for a in sorted(getattr(mic, "_pyroAttrs", [])):
    print(" -", a)

print("\n[SCAN/SIM Related]")
for m in sorted(mic._pyroMethods):
    if any(k in m.lower() for k in ["scan", "simulate", "sim", "speed", "rate", "parameter", "setting"]):
        print(" -", m)
        
uri = "PYRO:microscope.server@localhost:9092" #port for the AFM DT 9092
mic_server = Pyro5.api.Proxy(uri)

data_path = "h5_files"
file_name = "cufoil_high.h5"
mic_server.initialize_microscope("AFM", data_path=str((Path(data_path)/file_name).resolve()))
mic_server.setup_microscope(data_source = 'Compound_Dataset_1')
mic_server.get_dataset_info()

URI = "PYRO:microscope.server@localhost:9092"
H5_PATH = (Path("h5_files") / "cufoil_high.h5").resolve()
DATA_SOURCE = "Compound_Dataset_1"

def show_pyro_trace():
    print("".join(Pyro5.errors.get_pyro_traceback()))

def connect_ready():
    mic = Pyro5.api.Proxy(URI)
    mic._pyroBind()

    try:
        mic.initialize_microscope("AFM", data_path=str(H5_PATH))
        mic.setup_microscope(data_source=DATA_SOURCE)
        info = dict(mic.get_dataset_info())
        print("signals:", info.get("signals"))
        return mic, info
    except Exception:
        print("[FAIL] init/setup or get_dataset_info failed (server not ready).")
        show_pyro_trace()
        raise

mic, info = connect_ready()

print("scan_rate:", float(mic.scan_rate))
print("sample_rate:", float(mic.sample_rate))

uri = "PYRO:microscope.server@localhost:9092"
mic_server = Pyro5.api.Proxy(uri)
mic_server._pyroBind()

h5_path = (Path("h5_files") / "cufoil_high.h5").resolve()

mic_server.initialize_microscope("AFM", data_path=str(h5_path))
mic_server.setup_microscope(data_source="Compound_Dataset_1")

print(dict(mic_server.get_dataset_info()))

kwargs_low_I = {'I': 10, 'dz':5e-9, 'sample_rate': float(mic.sample_rate)}
mod_dict = [{'effect': 'real_PID', 'kwargs': kwargs_low_I}]
SCAN_RATE_FACTOR = 16


array_list, shape, dtype = mic_server.get_scan(
    channels=['HeightRetrace', 'Phase1Retrace'],
    modification=mod_dict,
    scan_rate=float(mic.scan_rate)*SCAN_RATE_FACTOR,
    direction='vertical',
    trace='backward'
)

dat = np.array(array_list, dtype=dtype).reshape(shape)
print(dat.shape)

CHANNEL_INDEX = 0          
LINE_FACTOR   = 4          
POINT_FACTOR  = 4          
MODE          = "decimate" 
FLIP_UD = True

SAVE          = True
OUT_DIR       = Path("DTM_data")
OUT_BASENAME  = f"HeightRetrace_HRLR_L{LINE_FACTOR}_P{POINT_FACTOR}_{MODE}"

def downsample_2d(img, fy, fx, mode=MODE):
    img = np.asarray(img)
    Ny, Nx = img.shape
    Ny2 = (Ny // fy) * fy
    Nx2 = (Nx // fx) * fx
    img_c = img[:Ny2, :Nx2] 
    if mode == "decimate":
        return img_c[::fy, ::fx]
    if mode == "bin":
        return img_c.reshape(Ny2//fy, fy, Nx2//fx, fx).mean(axis=(1, 3))
    raise ValueError("MODE must be 'decimate' or 'bin'")

def robust_limits(a, p_low=1, p_high=99):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None, None
    vmin, vmax = np.percentile(a, [p_low, p_high])
    if np.isclose(vmin, vmax):
        return None, None
    return float(vmin), float(vmax)

dat_arr = np.asarray(dat)
print("dat.shape:", dat_arr.shape)

if dat_arr.ndim == 2:
    height_hr = dat_arr.astype(np.float32, copy=False)
elif dat_arr.ndim == 3:
    if not (0 <= CHANNEL_INDEX < dat_arr.shape[0]):
        raise ValueError(f"CHANNEL_INDEX={CHANNEL_INDEX}가 dat의 채널 범위를 벗어났습니다. (C={dat_arr.shape[0]})")
    height_hr = dat_arr[CHANNEL_INDEX].astype(np.float32, copy=False)
else:
    raise ValueError(f"dat ndim={dat_arr.ndim} 지원 안 함. (2D 또는 3D만)")

height_lr = downsample_2d(height_hr, fy=LINE_FACTOR, fx=POINT_FACTOR, mode=MODE).astype(np.float32, copy=False)

def maybe_flip_ud(x):
    return np.flipud(x) if FLIP_UD else x

height_hr_out = maybe_flip_ud(height_hr)
height_lr_out = maybe_flip_ud(height_lr)


print("HR shape:", height_hr.shape)
print("LR shape:", height_lr.shape, f"(LINE_FACTOR={LINE_FACTOR}, POINT_FACTOR={POINT_FACTOR}, MODE={MODE})")

vmin, vmax = robust_limits(np.concatenate([height_hr.ravel(), height_lr.ravel()]))

fig, ax = plt.subplots(1, 2, figsize=(7, 3))

ax[0].imshow(height_hr_out, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
ax[0].set_title(f"HR HeightRetrace {height_hr.shape[0]}×{height_hr.shape[1]}")

ax[1].imshow(height_lr_out, origin="lower", cmap="cividis", vmin=vmin, vmax=vmax)
ax[1].set_title(f"LR HeightRetrace {height_lr.shape[0]}×{height_lr.shape[1]}")

for a in ax:
    a.set_xticks([])
    a.set_yticks([])

plt.tight_layout()
plt.show()

if SAVE:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = OUT_DIR / f"{file_name}_{float(mic.scan_rate)*SCAN_RATE_FACTOR}.npz"
    np.savez_compressed(
        npz_path,
        HR=height_hr,
        LR=height_lr,
        line_factor=int(LINE_FACTOR),
        point_factor=int(POINT_FACTOR),
        mode=str(MODE),
        channel_index=int(CHANNEL_INDEX) if dat_arr.ndim == 3 else -1,
    )

    np.save(OUT_DIR / f"{file_name}_{float(mic.scan_rate)*SCAN_RATE_FACTOR}_HR.npy", height_hr_out)
    np.save(OUT_DIR / f"{file_name}_{float(mic.scan_rate)*SCAN_RATE_FACTOR}_LR.npy", height_lr_out)