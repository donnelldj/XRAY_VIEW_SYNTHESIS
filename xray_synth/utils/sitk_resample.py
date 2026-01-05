from __future__ import annotations

import numpy as np
import SimpleITK as sitk

def resample_to_fixed_cube_preserve_extent(
    itk_img: sitk.Image,
    out_size_xyz: tuple[int, int, int] = (256, 256, 256),
) -> sitk.Image:
    """
    Resample to target voxel grid size (X,Y,Z) while preserving physical extent.
    Keeps origin/direction. No cropping.
    """
    in_size = np.array(list(itk_img.GetSize()), dtype=np.float64)        # (X,Y,Z)
    in_spacing = np.array(list(itk_img.GetSpacing()), dtype=np.float64)  # (sx,sy,sz)
    out_size = np.array(list(out_size_xyz), dtype=np.int64)

    extent = in_size * in_spacing
    out_spacing = extent / np.maximum(out_size.astype(np.float64), 1.0)

    resample = sitk.ResampleImageFilter()
    resample.SetSize([int(x) for x in out_size.tolist()])
    resample.SetOutputSpacing([float(x) for x in out_spacing.tolist()])
    resample.SetOutputDirection(itk_img.GetDirection())
    resample.SetOutputOrigin(itk_img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(-1024.0)
    return resample.Execute(itk_img)
