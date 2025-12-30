# 🔬 VantaScope 5090 Pro ⚛️
### Automated Computational Metrology for Advanced Materials Characterization

**VantaScope 5090 Pro** is a high-throughput characterization platform designed to bridge the gap between raw microscopy and quantitative materials science. Developed for the *Hack the Microscope* event at the University of Cincinnati, it utilizes a multi-stage neural architecture to provide real-time structural and electronic metrology from STEM, AFM, and STM micrographs.

---

## 🌟 Scientific Innovation
Current microscopy workflows often suffer from subjective manual interpretation and low throughput. VantaScope addresses these bottlenecks by integrating physics-informed AI:

* **DINOv2 Perception:** Self-supervised Vision Transformers extract rich visual features without the need for domain-specific labels.
* **Fuzzy Graph Attention Networks (Fuzzy-GAT):** Atomic structures are treated as graphs to model spatial relationships and defect interactions.
* **Physics-Informed Latent Space:** A disentangled autoencoder architecture aligns AI reasoning with unit cell geometry and topological invariants.
* **Uncertainty Quantification:** Bayesian methods decompose epistemic (model) and aleatoric (sensor) uncertainty for calibrated scientific decision-making.

---

## 🛠️ Key Functionality
* **Multi-Modal Ingestion:** Supports high-resolution file uploads (TIFF/PNG), real-time camera capture, and remote datastore URLs.
* **Digital Twin Reconstruction:** Side-by-side comparison of raw sensor data with AI-generated topological maps.
* **Quantitative Metrology:**
    * **Structural:** Lattice registry, defect density ($cm^{-2}$), and crystallinity metrics.
    * **Electronic:** Bandgap estimation ($eV$), conductivity classification, and predicted carrier mobility ($cm^2/Vs$).
* **Explainable AI (XAI):** Interactive attention heatmaps visualize the specific image patches driving property predictions.

---

## 📂 Project Structure
```text
vantascope/
├── src/vantascope/
│   ├── models/        # DINOv2 backbones, Fuzzy-GAT, and Uncertainty logic
│   ├── analysis/      # Property prediction and linguistic engines
│   ├── visualization/ # Plotly-based scientific overlays
│   └── interface/     # Streamlit Pro UI logic and components
├── data/              # Reference samples (CVD Graphene, SWCNT, GO)
├── requirements.txt   # Environment dependencies
└── README.md          # Project documentation
