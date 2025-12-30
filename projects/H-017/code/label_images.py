import pandas as pd
from joblib import Parallel, delayed
import numpy as np

import re

def compile_tag_patterns(tag_list):
    patterns = []
    for label, t in enumerate(tag_list):
        if isinstance(t, list):
            regex = "|".join(re.escape(x.lower()) for x in t)
            patterns.append((label, t[0], re.compile(regex)))
        else:
            patterns.append((label, t, re.compile(re.escape(t.lower()))))
    return patterns

def tag_chunk(df_chunk, patterns):
    captions = df_chunk['caption'].str.lower()

    df_chunk['tag'] = None
    df_chunk['label'] = None

    for label, tag_name, pattern in patterns:
        mask = captions.str.contains(pattern, regex=True, na=False)
        df_chunk.loc[mask & df_chunk['tag'].isna(), 'tag'] = tag_name
        df_chunk.loc[mask & df_chunk['label'].isna(), 'label'] = label

    return df_chunk.dropna(subset=['tag'])


def caption_labeller_parallel(DF, tag_list, n_jobs=-1, chunk_size=100_000):

    print("Commencing parallel labeling...")

    patterns = compile_tag_patterns(tag_list)

    chunks = np.array_split(DF, len(DF) // chunk_size + 1)

    processed = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(tag_chunk)(chunk.copy(), patterns)
        for chunk in chunks
    )

    DF_out = pd.concat(processed, ignore_index=True)

    # ---------- size dict ----------
    size_dict = (
        DF_out['tag']
        .value_counts()
        .to_dict()
    )

    max_val = max(size_dict.values())
    size_dict = {
        k: np.around(v * 20 / max_val + 7, 2)
        for k, v in size_dict.items()
    }

    sorted_dict = dict(sorted(size_dict.items(), key=lambda x: x[1]))

    print("Labelling complete.")
    return DF_out, sorted_dict

    

tag_list = [

    # ─────────────────────────────
    # Scanning Probe Microscopy (SPM)
    # ─────────────────────────────
    ['AFM', 'atomic force microscopy', 'AFM image', 'AFM topography',
     'phase imaging', 'contact mode', 'tapping mode', 'non-contact'],
    ['STM', 'scanning tunneling microscopy', 'tunneling current'],
    ['KPFM', 'Kelvin probe force microscopy', 'surface potential map'],
    ['MFM', 'magnetic force microscopy'],
    ['PFM', 'piezoresponse force microscopy'],
    ['C-AFM', 'conductive AFM', 'current mapping'],
    ['SNOM', 'NSOM', 'near-field scanning optical microscopy'],

    # ─────────────────────────────
    # Electron Microscopy – SEM / FIB
    # ─────────────────────────────
    ['SEM', 'scanning electron microscopy', 'SEM image', 'secondary electron',
     'backscattered electron', 'BSE micrograph'],
    ['EBSD', 'electron backscatter diffraction', 'orientation map',
     'inverse pole figure', 'IPF', 'grain orientation'],
    ['EDS', 'EDX', 'energy dispersive X-ray spectroscopy', 'elemental mapping'],
    ['CL', 'cathodoluminescence microscopy'],
    ['FIB', 'focused ion beam', 'FIB milling', 'cross-section'],
    ['FIB-SEM', 'serial sectioning', '3D reconstruction'],

    # ─────────────────────────────
    # Transmission Electron Microscopy (TEM)
    # ─────────────────────────────
    ['TEM', 'transmission electron microscopy'],
    ['HRTEM', 'high resolution TEM', 'lattice fringes'],
    ['STEM', 'scanning transmission electron microscopy'],
    ['HAADF', 'high-angle annular dark field', 'Z-contrast'],
    ['BF', 'bright field TEM'],
    ['DF', 'dark field TEM'],
    ['SAED', 'SAD', 'selected area electron diffraction'],
    ['CBED', 'convergent beam electron diffraction'],
    ['EELS', 'electron energy loss spectroscopy'],
    ['STEM-EDS', 'atomic resolution mapping'],
    ['in-situ TEM', 'heating TEM', 'biasing TEM'],

    # ─────────────────────────────
    # Optical & Confocal Microscopy
    # ─────────────────────────────
    ['Optical microscopy', 'optical micrograph', 'optical photograph'],
    ['Polarized optical microscopy', 'POM'],
    ['Confocal microscopy', 'laser scanning confocal microscopy'],
    ['Fluorescence microscopy'],
    ['Phase contrast microscopy'],
    ['Dark field optical microscopy'],
    ['Interference microscopy'],

    # ─────────────────────────────
    # Microstructure & Morphology
    # ─────────────────────────────
    ['microstructure', 'microstructural evolution'],
    ['grain size', 'grain size distribution', 'GSD'],
    ['particle size', 'particle size distribution', 'PSD'],
    ['porosity', 'pore size distribution'],
    ['agglomeration', 'clustering'],
    ['surface roughness', 'Ra', 'RMS roughness'],
    ['fractography', 'fracture surface'],
    ['crack', 'crack propagation', 'crack deflection'],
    ['interface', 'interfacial region', 'interphase'],
    ['coating thickness', 'film thickness'],
    ['columnar grains', 'equiaxed grains'],

    # ─────────────────────────────
    # Crystallography & Image Analysis
    # ─────────────────────────────
    ['lattice fringe', 'interplanar spacing', 'd-spacing'],
    ['lattice parameter', 'lattice constant'],
    ['FFT', 'fast Fourier transform', 'FFT pattern'],
    ['moire pattern', 'moire fringes'],
    ['defects', 'dislocations', 'stacking fault'],
    ['twins', 'grain boundary', 'sub-grain boundary'],
    ['orientation relationship'],
    ['phase contrast'],

    # ─────────────────────────────
    # 3D / Advanced Microscopy
    # ─────────────────────────────
    ['3D microscopy', '3D microstructure'],
    ['electron tomography', 'TEM tomography'],
    ['X-ray microscopy', 'X-ray micro-CT', 'nano-CT'],
    ['serial sectioning'],
    ['correlative microscopy', 'multi-modal microscopy'],

    # ─────────────────────────────
    # Imaging & Mapping Terms
    # ─────────────────────────────
    ['elemental map', 'composition map'],
    ['current map'],
    ['strain mapping'],
    ['orientation mapping'],
    ['phase mapping'],
    ['contrast mechanism'],
    ['image segmentation'],
    ['image processing']
]

take_tags = ['TEM','SEM','EDS','FIB','AFM','microstructure','EBSD','Optical microscopy','STM','PFM','MFM']

df = pd.read_csv('../data/corpus_acta_mini.csv')

newdf, res = caption_labeller_parallel(df, tag_list, n_jobs=4)
new_mask = newdf.tag.isin(take_tags)
newdf2 = newdf[new_mask]

print(newdf2.shape)
print('Total images: ', df.shape[0])
print('Total images with tags: ', newdf2.shape[0])
print('Tags summary: ', newdf2.tag.value_counts())
print('Tagged data saved to ../data/corpus_acta_mini_tagged.csv')
newdf2.to_csv('../data/corpus_acta_mini_tagged.csv', index=False)



