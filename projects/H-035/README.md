# Microscopy Hackathon 2025 Team LASER

## Angle-Dependent Morphologies of Ferroelectric Domain Walls

_Laurel Washburn, Ehtesham Anwar, Sho Watanabe, Aidan Cotton, Kaurab Gautam, Nabin Khanal, Ralph Bulanadi_

## Notes:

* Code used to acquire Auto-3DPFM data, and calculate DW position was previously submitted for publication in [1].
* ResHedNet neural network was based on the implementation published in [2].
* Files Image.hf5.zip, Image2.hf5.zip, Image3.hf5.zip, Image_1V.hf5.zip, Image_rot90.hf5.zip contain raw 3DPFM data
* ResHedNet_DWs_Auto3DPFM.ipynb was adapted from Ref. [2] by Ralph Bulanadi as a neural network to analyse Auto-3DPFM data
* 3DPFM_Analysis_Centerline-Width.ipynb was developed by Ralph Bulanadi to characterize domain wall sinuosity and thickness
* Spearman_Correlation_3DPFM_Analysis.ipynb was developed by Ehtesham Anwar to analyze domain walls and topography
* 3DPFM_Analysis_DW_HoughLines.ipynb was developed by Laurel Washburn to fit lines to domain wall positions

## Introduction

Ferroelectric materials maintain a spontaneous electric polarization that is often associated with a mechanical strain in the same axis. The interplay of both electrostatic and mechanical boundary conditions lead to the emergence of ferroelectric domains—regions dominated by a particular ferroelectric orientation—bounded by ferroelectric domain walls (DWs).

The properties of such DWs, such as writability, increased conductivity, or mechanical softness yield immense potential for future applications in nanoelectronics, but the complexity of their formation and structure mandate extensive further research before these applications can be realized. The system may relax to minimize charge buildup from adjacent domains at DWs, for example, yielding particular preferred DW structures associated with the vector-angle difference of its two bounding domains. The elastic boundary conditions of single grains or single crystals may also restrict domain-wall conformations: In rhombohedral ferroelectrics, such as zirconium-rich PZT, the ferroelectric polarization forms along the {111} axes; as the vector-angle difference between any two {111} vectors are either 71°, 109°, or 180°, the DWs between these domains likewise forms these characteristic angles. This interplay between the crystallographic constraints of domains, and the minimization of the interfacial energy of ferroelectric DW, yield complex patterns and properties. An enhanced understanding of this DW-structure–property relationship can therefore yield new strides in the development of nanoelectronic devices.

Fully automated three-dimensional piezoresponse force microscopy (Auto-3DPFM) has recently allowed for more accessible quantitative identification of imaged ferroelectric DWs [1], and analysis of this data can lead new insights into how the characteristic angle of a DW can affect its shape; however, these datasets, mapping two-dimensional images of complex three-dimensional vectors, can be obtrusive to standard analysis techniques. As such, we here leverage artificial-intelligence and machine learning methods to identify and characterise DW shape against their characteristic angle, pushing forward our understanding of DW structure and applications.

## Results

_**Distinguishing DW Angle by Neural Networks:**_ To verify that the characteristic angle of a DW affects its shape, we first used ResHedNet [2], a neural network architecture designed to identify the position of DWs from a PFM map of ferroelectric polycrystalline PZT. When ResHedNet was trained on four experiments of Auto-3DPFM data, augmented with an 80/20 training/testing split, and with all ferroelectric DWs marked and annotated, the neural network was able to identify all DWs of a new dataset. However, when ResHedNet was trained only on images of domain-walls with a characteristic angle θ > 130°, ResHedNet was successfully able to identify only these large-vector-angle-difference ferroelectric DWs. Likewise, ResHedNet trained only on DWs with a characteristic angle 20° < θ < 90° could only identify low-angle ferroelectric DWs. Such a process proves the capability for a well-trained neural network system to not only identify DWs from Auto-3DPFM data, but further, distinguish them by their characteristic angle.

_**Computer Vision and Ferroelectric DWs: Hough Lines and Sinuosity:**_ While the neural network shows the capability to distinguish these DWs, they did not themselves provide insight into the process by which images of DWs can be distinguished. To further investigate the DW images, we used computer-vision methods.

The probabilistic Hough Line Transform in OpenCV [3] was used to detect straight lines, and these were found to be associated with ferroelectric DWs of low and intermediate characteristic angles. However, the thickness, roughness, and complexity of the domain walls made it difficult to cleanly isolate the walls of interest from the 3DPFM images, and multiple overlapping lines were detected for each wall. A function to merge duplicate lines together was developed, but more work is needed to fine tune the image processing procedure and line detection algorithm before it can be reliably used for domain wall identification and analysis.

Meanwhile, the sinuosity—the ratio between the length of a curve, and the distance between its two edges—were also calculated for different DW angles based on the centerline-width [4] implementation. For 20° < θ < 90°, the sinuosity was calculated to be 1.1 ± 0.3, while for θ > 90°, sinuosity was higher at 1.2 ± 0.5. Tools such as sinuosity calculation of line detection pave the way for continued analysis of ferroelectric DWs.

_**Investigation of Mechanical Properties by Spearman Correlation:**_ Finally, to establish the utility of DW characterisation, we examined mechanical characteristics  associated with each ferroelectric domain wall measured from Auto-3DPFM data. Throughout the sample, a strong positive correlation (ρ ≈ 0.56) is observed between variation in local height and roughness, as expected for such metrics. In contrast, local topographical curvature (defined as the sum of the second spatial derivative in each direction) shows negligible correlation with either thickness (ρ ≈ 0.01) or roughness (ρ ≈ −0.04).

When mapped against particular DW orientations, DWs with a higher characteristic angle (above approximately 100 degrees) express increases in roughness, along with an increase in variability of topographical curvature. Compared to their surrounding domains, domain walls in general present an increase in local roughness, and an increase in range in topographical curvature.

## Discussion and Conclusions:
Through this work, we’ve investigated the structure and properties of ferroelectric DWs imaged by Auto-3DPFM. While Auto-3DPFM has previously shown a variety of different DW orientations, the integration of the neural network ResHedNet proved the potential to differentiate between different DW orientations by their appearance alone. Through computer vision by Hough lines, and through calculation of DW sinuosity, we have shown that lower-angle DWs tend to be more straight than their counterparts with a higher angle.

We again attribute this behaviour to the interplay between electric and mechanical boundary conditions. In PZT, the ferroelectric polarization is accompanied with a strain, but across a 180° DW, where the polarization is antiparallel, there is corresponding elastic response. Meanwhile, in lower-angle domain-walls, where the DWis accompanied with an elastic response, the DW is more restricted by a material’s crystallographic axis and therefore a straighter DW becomes preferable. These descriptors, when coupled with continued research into the material properties of ferroelectric DWs, can pave the way for new research and the development of new DW devices.

## References:

[1] Bulanadi, Ralph, et al. "Auto-3DPFM: Automating Polarization-Vector Mapping at the Nanoscale." arXiv preprint arXiv:2512.09249 (2025).

[2] Liu, Yongtao, et al. "Exploring physics of ferroelectric domain walls in real time: deep learning enabled scanning probe microscopy." Advanced Science 9.31 (2022): 2203957.

[3] Bradski, Gary. “The OpenCV Library.” Dr. Dobb’s Journal of Software Tools, 2000.

[4] cyschneck. centerline-width. Version 2.0.1, PyPI, 24 June 2025, https://pypi.org/project/centerline-width/. PyPI



