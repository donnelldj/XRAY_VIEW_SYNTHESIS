# import streamlit as st
# import SimpleITK as sitk
# import numpy as np

# st.set_page_config(layout="wide", page_title="X-Ray Synthesis Partner")

# def normalize_for_display(image_array):
#     """Normalizes an array to 0.0 - 1.0 range for Streamlit display."""
#     img_min = np.min(image_array)
#     img_max = np.max(image_array)
#     if img_max - img_min == 0:
#         return image_array
#     return (image_array - img_min) / (img_max - img_min)

# @st.cache_data
# def process_medical_data(path):
#     # 1. Load the volume
#     itk_img = sitk.ReadImage(path)
    
#     # 2. Resample to 1mm Isotropic (Crucial for geometric accuracy)
#     original_spacing = itk_img.GetSpacing()
#     original_size = itk_img.GetSize()
#     new_spacing = [1.0, 1.0, 1.0]
#     new_size = [
#         int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
#         for i in range(3)
#     ]
    
#     resample = sitk.ResampleImageFilter()
#     resample.SetOutputSpacing(new_spacing)
#     resample.SetSize(new_size)
#     resample.SetOutputDirection(itk_img.GetDirection())
#     resample.SetOutputOrigin(itk_img.GetOrigin())
#     resample.SetInterpolator(sitk.sitkLinear)
#     itk_img_resampled = resample.Execute(itk_img)
    
#     # 3. Get Array (Hounsfield Units)
#     data = sitk.GetArrayFromImage(itk_img_resampled)
    
#     # 4. Synthesize X-rays (Mean Intensity Projection)
#     # Shift HU so air is approx 0 and bones are high
#     attenuation_data = data + 1024 
#     attenuation_data[attenuation_data < 0] = 0
    
#     # Projections (Mean along different axes)
#     # We flip them vertically (axis 0) so the anatomy isn't upside down
#     xray_ap = np.flipud(np.mean(attenuation_data, axis=1))    # Coronal projection
#     xray_lat = np.flipud(np.mean(attenuation_data, axis=2))   # Sagittal projection
    
#     # Normalize projections for st.image
#     xray_ap_norm = normalize_for_display(xray_ap)
#     xray_lat_norm = normalize_for_display(xray_lat)
    
#     return data, xray_ap_norm, xray_lat_norm

# # --- UI ---
# st.title("🩻 Research-Grade X-Ray Synthesis")

# # Ensure this path is correct for your local environment
# file_path = r"data\luna16\subset0\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"

# try:
#     volume, xray_ap, xray_lat = process_medical_data(file_path)
    
#     tab1, tab2 = st.tabs(["3D Volume Slicer", "Synthesized X-Rays (DRR)"])
    
#     with tab1:
#         st.subheader("Anatomically Correct Slicing (1.0mm Isotropic)")
#         z, y, x = volume.shape
#         c1, c2, c3 = st.columns(3)
        
#         with c1:
#             idx_z = st.slider("Axial Slice", 0, z-1, z//2)
#             # Clip HU for better soft tissue/bone contrast (-1000 to 400 is standard CT window)
#             st.image(normalize_for_display(np.clip(volume[idx_z, :, :], -1000, 400)), 
#                      use_container_width=True, caption="Axial View")
#         with c2:
#             idx_y = st.slider("Coronal Slice", 0, y-1, y//2)
#             # Flip Coronal and Sagittal for upright viewing
#             cor_slice = np.flipud(volume[:, idx_y, :])
#             st.image(normalize_for_display(np.clip(cor_slice, -1000, 400)), 
#                      use_container_width=True, caption="Coronal View")
#         with c3:
#             idx_x = st.slider("Sagittal Slice", 0, x-1, x//2)
#             sag_slice = np.flipud(volume[:, :, idx_x])
#             st.image(normalize_for_display(np.clip(sag_slice, -1000, 400)), 
#                      use_container_width=True, caption="Sagittal View")

#     with tab2:
#         st.subheader("Digitally Reconstructed Radiographs (DRR)")
#         st.info("Simulated X-ray via Mean Intensity Projection (MIP).")
        
#         col_ap, col_lat = st.columns(2)
        
#         with col_ap:
#             st.write("**AP View (Anteroposterior)**")
#             # Displaying the normalized projection
#             st.image(xray_ap, use_container_width=True, caption="Frontal Projection")
            
#         with col_lat:
#             st.write("**Lateral View (Side Profile)**")
#             st.image(xray_lat, use_container_width=True, caption="Side Projection")

# except Exception as e:
#     st.error(f"Error loading data: {e}")
#     st.warning("Check if the file path to the .mhd file is correct.")