# Aiscia Foam Stability Simulator

# Getting Started (Simulator Runbook)

# Download data

Grab the Drive folder from https://drive.google.com/drive/folders/1yv3cSJiI7Sge4IkjKMYtp4lowY44Q0A7?usp=drive_link.
Place its contents at: /DTMicroscope/data/foam/
(you should see H5 files like 1_wt_capryl_glucoside.h5, nanoparticle_0_025_wt.h5, etc.)
Required training table

Ensure /DTMicroscope/notebooks/foam/foam_training_table_with_nano.csv exists.
This CSV must include inputs: time_s, surfactant wt.% columns, nanoparticle_al2o3_wt_pct (0 for legacy).
Outputs: bubble_count_per_mm², mean/SD bubble area, radii (Ravg/Rrms/R21/R32), v/w, lamella_thickness_mm, half_life_s.
Train the model (if needed).

# Run the simulator

Open notebooks/foam/foam_inputs_to_image_with_model.ipynb.
It will auto-detect data/foam H5s and load foam_training_table_with_nano.csv + foam_rfr.pkl.
Run cells to:
Predict targets from your inputs.
Map to nearest real frames (composition-aware, nearest-time variants).
Generate simulated frames and GIFs:
time_sweep.gif (time sweep for a composition)
time_sweep_compare.gif (actual vs simulated side-by-side)
time_sweep_compare_all.gif, time_sweep_real.gif (alternative comparisons)


H5 frames: AisciaMicroscopeHackathon/DTMicroscope/data/foam/
Training table: AisciaMicroscopeHackathon/DTMicroscope/notebooks/foam/foam_training_table_with_nano.csv
Trained model: AisciaMicroscopeHackathon/DTMicroscope/notebooks/foam/foam_rfr.pkl
Simulator notebook: AisciaMicroscopeHackathon/DTMicroscope/notebooks/foam/foam_inputs_to_image_with_model.ipynb
