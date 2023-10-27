from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import numpy as np

'''
Load the data from a list of paths where the images are stored.
'''
class CardiacImageDataset(Dataset):

  def __init__(self, image_paths, transform=None):
    """
    Arguments:
      data(list or tuple): The list with the paths were the images are stored
      transform(callable, optional): Optional transform to be applied 
      on a sample
    """
    # check if the image paths are a list or tuple
    assert isinstance(image_paths, (list, tuple))

    # safe the image paths
    self.image_paths = image_paths
    # safe the transform
    self.transform = transform

  def __len__(self):
    # return the length of the dataset
    return len(self.image_paths)

  def __getitem__(self, idx):
    # read the image
    image = sitk.ReadImage(self.image_paths[idx])
    image_array = sitk.GetArrayFromImage(image)
    # on dictionary
    sample = {"image": image, "labels": np.unique(image_array)}
    # if there is transform,apply
    if self.transform:
      sample = self.transform(sample)
    return sample

class Rescale(object):
  '''
  Rescale a sample to a given size
  '''
  def __init__(self, output_size, output_spacing, method="KNearestNeightbor"):
    '''
    Arguments:
    output_size (list or tuple): The volume size that is desire to rescale
    it is better when the axis (x,y) have the same size, i.e.(256,256)

    output_spacing(list or tuple): The desire spacing for the image. Its is
    important to have a new spacing because it is useful for the factor scale

    method (string): The desire sitk method for the images
    '''
    # check if the output size is a list or tuple
    assert isinstance(output_size, (list, tuple))
    self.output_size = output_size

    assert isinstance(output_spacing, (list, tuple))
    self.output_spacing = output_spacing

    assert isinstance(method, str)
    self.method = method

  def __call__(self, sample):
    image = sample["image"]

    # Define the desired output volume size as an array
    VolSize = np.asarray(self.output_size, dtype=int)

    # Set the interpolation method to 'KNearestNeightbor' or 'Linear'
    if self.method == "KNearestNeightbor":
      method = sitk.sitkNearestNeighbor
    elif self.method == "Linear":
      method = sitk.sitkLinear
    else:
      raise ValueError("The method is not valid")

    # Define the desired output voxel spacing as an array [1.5625, 1.5625, 10.0]
    dstRes = np.asarray(self.output_spacing, dtype=float)

    # Create an empty numpy array to store the result with the specified dimensions
    ret = np.zeros([VolSize[0], VolSize[1], VolSize[2]], dtype=np.float32)

    # Get the spacing (distance between voxels) along each axis (x, y, z) from the input image
    x_mm, y_mm, z_mm = image.GetSpacing()

    # Get the dimensions (number of voxels) of the input image along each axis (rows, columns, slices)
    r, c, s = image.GetSize()

    # Calculate the scaling factor to resample the image based on the desired spacing
    factor = np.asarray(image.GetSpacing()) / [dstRes[0], dstRes[1], dstRes[2]]

    # Calculate the new size of the image after resampling
    factorSize = np.asarray(image.GetSize() * factor, dtype=float)

    # Get the new dimensions after resampling (rows, columns, slices)
    r_new, c_new, s_new = factorSize

    # Calculate the final new size as the maximum of factorSize and VolSize
    newSize = np.max([factorSize, VolSize], axis=0)

    # Convert the new size to integer values and create a list
    newSize = newSize.astype(dtype=int).tolist()

    # Create an identity 3D affine transformation
    T = sitk.AffineTransform(3)

    # Set the transformation matrix of T to be the same as the input image's direction matrix
    T.SetMatrix(image.GetDirection())

    # Create a resampling filter
    resampler = sitk.ResampleImageFilter()

    # Set the reference image for resampling as the input image
    resampler.SetReferenceImage(image)

    # Set the output voxel spacing for resampling
    resampler.SetOutputSpacing([dstRes[0], dstRes[1], dstRes[2]])

    # Set the size of the output image after resampling
    resampler.SetSize(newSize)

    # Set the interpolation method for resampling
    resampler.SetInterpolator(method)

    # Execute the resampling on the input image
    imgResampled = resampler.Execute(image)

    # Calculate the centroid (center) of the new image in world coordinates
    imgCentroid = np.asarray(newSize, dtype=float) / 2.0

    # Calculate the starting pixel coordinates for cropping the resampled image
    imgStartPx = (imgCentroid - VolSize / 2.0).astype(dtype=int)

    # Create a region of interest filter
    regionExtractor = sitk.RegionOfInterestImageFilter()

    # Set the size of the region of interest (ROI) as VolSize
    regionExtractor.SetSize(VolSize.astype(dtype=int).tolist())

    # Set the starting index of the ROI based on imgStartPx
    regionExtractor.SetIndex(imgStartPx.tolist())

    # Execute the ROI extraction on the resampled image
    imgResampledCropped = regionExtractor.Execute(imgResampled)

    # Transpose the image data to match the desired orientation (x, y, z) and store it in the ret variable
    ret = np.transpose(sitk.GetArrayFromImage(
        imgResampledCropped).astype(dtype=float), [1, 2, 0])

    return {"image": ret, "labels": sample["labels"]}


class Outliers(object):
  '''
  It will remove the outliers from the image replacing the upper outliers with mean + 3stdDev
  '''

  def __call__(self, sample):
    # Define the image type

    image = sample["image"]

    # Get the mean value from the entire image
    mean_value = np.mean(image)

    # Get the standard deviation from the entire image
    stdDev_value = np.std(image)

    # Get the upper threshold for the pixels
    upper_threshold = mean_value + 3 * stdDev_value

    # Change the values from the outliers by this upper threshold value
    image[image > upper_threshold] = upper_threshold

    return {"image": image, "labels": sample["labels"]}


class Normalization(object):
  '''
  It will normalize the image values with the Min-Max Scaling Method
  '''

  def __call__(self, sample):
    image = sample["image"]
    labels = sample["labels"]

    # do a copy of the image
    normalize_image = np.zeros_like(image, dtype=np.float32)

    # the max value from the image that is a skit.image object
    max_value = np.max(image)
    # the min value from the image that is a skit.image object
    min_value = np.min(image)

    normalize_image = (image - min_value) / (max_value - min_value)

    return {"image": normalize_image}


class ToTensor(object):
  '''
  Convert ndarrays in sample to Tensors.
  '''
  def __call__(self, sample):
    image = sample["image"]
    image = image.transpose((2, 0, 1))  # Transpose to (C, H, W)
    return {"image": torch.from_numpy(image)}
