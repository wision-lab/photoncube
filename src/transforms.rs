use std::{
    convert::{From, Into},
    ops::BitOrAssign,
};

use ab_glyph::{FontRef, PxScale};
use anyhow::{anyhow, Result};
use clap::ValueEnum;
use fastrand;
use image::{imageops, GrayImage, ImageBuffer, Luma, Pixel, Rgb, RgbImage};
use imageproc::{
    definitions::{Clamp, Image},
    drawing::{draw_text_mut, text_size},
    map::map_pixels,
};
use ndarray::{
    concatenate, s, Array, Array2, Array3, ArrayBase, ArrayD, ArrayView2, ArrayView3, ArrayViewD,
    Axis, Data, IxDyn, NewAxis, Slice,
};
use ndarray_stats::{interpolate::Linear, QuantileExt};
use noisy_float::types::n64;
use numpy::Ix2;
use pyo3::prelude::*;
use strum_macros::EnumString;

#[pyclass(eq, eq_int)]
#[derive(ValueEnum, Clone, Copy, Debug, EnumString, PartialEq)]
pub enum Transform {
    Identity,
    Rot90,
    Rot180,
    Rot270,
    FlipUD,
    FlipLR,
}

#[pymethods]
impl Transform {
    /// Get transform from it's string repr, options are:
    /// "identity", "rot90", "rot180", "rot270", "flip-ud", "flip-lr"
    #[staticmethod]
    #[pyo3(name = "from_str", signature=(transform_name))]
    pub fn from_str_py(transform_name: &str) -> Result<Self> {
        Self::from_str(transform_name, true).map_err(|e| anyhow!(e))
    }
}

/// Expects Array of shape HW(C)
/// Cannot unpack along channel dimension
pub fn unpack_single<T>(bitplane: &ArrayViewD<'_, u8>, axis: usize) -> Result<ArrayD<T>>
where
    T: From<u8> + Copy + num_traits::Zero,
{
    let mut new_dim: Vec<usize> = bitplane.shape().into();
    new_dim[axis] *= 8;

    // Allocate full sized frame
    let mut unpacked_bitplane = ArrayD::<T>::zeros(IxDyn(&new_dim));

    // Iterate through slices with stride 8 of the full array and fill it up
    // Note: We reverse the shift to account for endianness
    for shift in 0..8 {
        let ishift = 7 - shift;
        let mut slice =
            unpacked_bitplane.slice_axis_mut(Axis(axis), Slice::from(shift..).step_by(8));
        let bit = (bitplane >> ishift) & 1;
        slice.assign(&bit.mapv(|i| T::from(i)));
    }

    Ok(unpacked_bitplane)
}

/// Expects Array of shape HW(C)
/// Cannot pack along channel dimension
pub fn pack_single(bitplane: &ArrayViewD<'_, u8>, axis: usize) -> Result<ArrayD<u8>> {
    let mut new_dim: Vec<usize> = bitplane.shape().into();
    new_dim[axis] = (new_dim[axis] as f32 / 8.0).ceil() as usize;

    // Allocate packed frame
    let mut packed_bitplane = ArrayD::<u8>::zeros(IxDyn(&new_dim));

    // Iterate through slices with stride 8 of the full array and fill up packed frame
    // Note: We reverse the shift to account for endianness
    for shift in 0..8 {
        let ishift = 7 - shift;
        let slice = bitplane.slice_axis(Axis(axis), Slice::from(shift..).step_by(8));
        let bit = slice.mapv(|v| (v & 1) << ishift);
        let h = slice.len_of(Axis(0));
        let w = slice.len_of(Axis(1));
        packed_bitplane
            .slice_axis_mut(Axis(0), Slice::from(..h))
            .slice_axis_mut(Axis(1), Slice::from(..w))
            .bitor_assign(&bit);
    }

    Ok(packed_bitplane)
}

// Replaces `nshare::ToImageLuma` (which isn't actually implemented?!)
pub fn array2_to_grayimage<T>(frame: Array2<T>) -> ImageBuffer<Luma<T>, Vec<T>>
where
    T: image::Primitive,
{
    ImageBuffer::<Luma<T>, Vec<T>>::from_raw(
        frame.len_of(Axis(1)) as u32,
        frame.len_of(Axis(0)) as u32,
        frame.into_raw_vec_and_offset().0,
    )
    .unwrap()
}

// Replaces `nshare::ToNdarray2`
pub fn grayimage_to_array2<T>(im: ImageBuffer<Luma<T>, Vec<T>>) -> Array2<T>
where
    T: image::Primitive,
{
    Array2::from_shape_vec((im.height() as usize, im.width() as usize), im.into_raw()).unwrap()
}

// Alternative to `nshare::RefNdarray2` which returns HW array
pub fn ref_grayimage_to_array2<T>(im: &ImageBuffer<Luma<T>, Vec<T>>) -> ArrayView2<'_, T>
where
    T: image::Primitive,
{
    ArrayView2::from_shape((im.height() as usize, im.width() as usize), im).unwrap()
}

// Given an NDarray of HxWxC, convert it to an RGBImage (C must equal 3)
// Note: Contrary to link below, we use HWC *not* CHW.
//       This means we cannot use `nshare::ToNdarray3`
//       as it converts an image to CHW format.
// https://stackoverflow.com/questions/56762026/how-to-save-ndarray-in-rust-as-image
pub fn array3_to_image<P>(arr: Array3<P::Subpixel>) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    P: image::Pixel,
{
    assert!(arr.is_standard_layout());
    let (height, width, _) = arr.dim();
    let (raw, _) = arr.into_raw_vec_and_offset();

    ImageBuffer::<P, Vec<P::Subpixel>>::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}

// Alternative to `nshare::ToNdarray3` which returns HWC array
pub fn image_to_array3<P>(im: ImageBuffer<P, Vec<P::Subpixel>>) -> Array3<P::Subpixel>
where
    P: image::Pixel,
{
    Array3::from_shape_vec(
        (
            im.height() as usize,
            im.width() as usize,
            P::CHANNEL_COUNT as usize,
        ),
        im.into_raw(),
    )
    .unwrap()
}

// Alternative to `nshare::RefNdarray3` which returns HWC array
pub fn ref_image_to_array3<P>(im: &ImageBuffer<P, Vec<P::Subpixel>>) -> ArrayView3<'_, P::Subpixel>
where
    P: image::Pixel,
{
    ArrayView3::from_shape(
        (
            im.height() as usize,
            im.width() as usize,
            P::CHANNEL_COUNT as usize,
        ),
        im,
    )
    .unwrap()
}

pub fn gray_to_rgbimage(frame: &GrayImage) -> RgbImage {
    // Convert it to rgb by duplicating each pixel
    map_pixels(frame, |_x, _y, p| Rgb([p[0], p[0], p[0]]))
}

pub fn annotate<P>(frame: &mut Image<P>, text: &str, color: P)
where
    P: Pixel,
    <P as Pixel>::Subpixel: Clamp<f32>,
    f32: From<<P as Pixel>::Subpixel>,
{
    let font = FontRef::try_from_slice(include_bytes!("DejaVuSans.ttf")).unwrap();
    let scale = PxScale { x: 20.0, y: 20.0 };

    draw_text_mut(frame, color, 5, 5, scale, &font, text);
    text_size(scale, &font, text);
}

pub fn apply_transforms<P>(frame: Image<P>, transform: &[Transform]) -> Image<P>
where
    P: Pixel + 'static,
{
    // Note: if we don't shadow `frame` as a mut, we cannot override it in the loop
    let mut frame = frame;

    for t in transform.iter() {
        frame = match t {
            Transform::Identity => continue,
            Transform::Rot90 => imageops::rotate90(&frame),
            Transform::Rot180 => imageops::rotate180(&frame),
            Transform::Rot270 => imageops::rotate270(&frame),
            Transform::FlipUD => imageops::flip_vertical(&frame),
            Transform::FlipLR => imageops::flip_horizontal(&frame),
        };
    }
    frame
}

/// Convert from linear colorspace to sRGB one.
/// Note: This is only well behaved for images in the `[0, 1]` range, so while it is
///     generic over T, this should be f32 or similar. This will not work over ints.
///
/// See: https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
pub fn linearrgb_to_srgb<T>(frame: &ArrayViewD<T>) -> ArrayD<T>
where
    T: Into<f32> + Clamp<f32> + Copy,
{
    frame.mapv(|x| {
        let x = T::into(x);
        if x < 0.0031308 {
            if x < 0.0 {
                return T::clamp(0.0);
            } else {
                return T::clamp(x * 12.92);
            }
        }
        T::clamp(1.055 * x.powf(1.0 / 2.4) - 0.055)
    })
}

/// Invert the process by which binary frames are simulated. The result can be either
/// linear RGB values or sRGB values depending on how the binary frames were constructed.
/// Assuming each binary patch was created by a Bernoulli process with `p=1-exp(-factor*rgb)`,
/// then the average of binary frames tends to `p`. We can therefore recover the original rgb
/// values as `-log(1-bin)/factor`.
///
/// Note: Again, we are generic over `T` but this should be a floating type.
pub fn binary_avg_to_rgb<T>(frame: &ArrayViewD<T>, factor: f32, quantile: Option<f64>) -> ArrayD<T>
where
    T: Into<f32> + Clamp<f32> + Copy,
{
    let mut frame = frame.mapv(|v| -(1.0 - T::into(v)).clamp(1e-6, 1.0).ln() / factor);

    if let Some(quantile_val) = quantile {
        let val = Array::from_iter(frame.iter().cloned())
            .quantile_axis_skipnan_mut(Axis(0), n64(quantile_val), &Linear)
            .unwrap()
            .mean()
            .unwrap();
        frame.par_mapv_inplace(|v| (v / val).clamp(0.0, 1.0));
    } else {
        // TODO: Is this branch needed?
        frame.par_mapv_inplace(|v| v.clamp(0.0, 1.0));
    };

    frame.mapv(|v| T::clamp(v))
}

// Note: This does not yet support full array, only TOP half.
pub fn process_colorspad<T>(frame: Array2<T>) -> Array2<T>
where
    T: Clone,
{
    let (h, _) = frame.dim();
    if h > 256 {
        unimplemented!("Full array processing not yet supported in `process_colorspad`.")
    }

    // Crop dead regions around edges
    let mut frame = frame.to_owned();
    let mut crop = frame.slice_mut(s![2.., ..496]);

    // Swap rows (Can we do this inplace?)
    let (mut slice_a, mut slice_b) = crop.multi_slice_mut((s![.., 252..256], s![.., 260..264]));
    let tmp_slice = slice_a.to_owned();
    slice_a.assign(&slice_b);
    slice_b.assign(&tmp_slice);

    // This clones the array, can we avoid this somehow??
    crop.to_owned()
}

/// Process raw grayspad frame, either 256x512 (top) or 512x512 is expected.
/// In both cases, the leftmost side is cropped.
#[allow(clippy::reversed_empty_ranges)]
pub fn process_grayspad<T>(frame: Array2<T>) -> Array2<T>
where
    T: Clone + num_traits::Zero,
{
    let (h, _) = frame.dim();

    if h == 512 {
        let top = frame.slice(s![..256, ..]);
        let btm = frame.slice(s![256.., ..]);
        let frame = concatenate(Axis(0), &[top, Array2::zeros((1, 512)).view(), btm]).unwrap();
        frame.slice(s![2..-2, ..496]).to_owned()
    } else if h == 256 {
        frame.slice(s![2.., ..496]).to_owned()
    } else {
        unimplemented!(
            "A frame of either 256x512 (top) or 512x512 is expected for `process_grayspad`."
        )
    }
}

// Note: The use of generics here is heavy handed, we only really want this function
//       to work with T=u8 or maybe T=f32/i32. Is there a better way? I.e generic over primitives?
// Supports dynamically sized arrays of 2 or 3 dims as long as the first two dimensions are HW
pub fn interpolate_where_mask<S1, S2, T>(
    frame: &ArrayBase<S1, IxDyn>,
    mask: &ArrayBase<S2, Ix2>,
    dither: bool,
) -> Result<ArrayD<T>>
where
    S1: Data<Elem = T>,
    S2: Data<Elem = bool>,
    T: Into<f32> + Clamp<f32> + Copy,
{
    let (frame, needs_squeeze) = if frame.ndim() == 2 {
        (
            frame.slice(s![.., .., NewAxis]).into_dimensionality()?,
            true,
        )
    } else if frame.ndim() == 3 {
        (frame.view().into_dimensionality()?, false)
    } else {
        return Err(anyhow!(
            "Expected frame to be Hw or HWC but got {} dimensions.",
            frame.ndim()
        ));
    };

    let (h, w, _c) = frame.dim();
    let (mask_h, mask_w) = mask.dim();

    if (mask_h < h) || (mask_w < w) {
        return Err(anyhow!(
            "Frame has size {:?} but interpolation mask has size {:?}.",
            frame.dim(),
            mask.dim()
        ));
    }

    let mut output = frame.to_owned();
    output.indexed_iter_mut().for_each(|((i, j, k), out_val)| {
        if mask[(i, j)] {
            let mut counter: f32 = 0.0;
            let mut value: f32 = 0.0;

            for ki in [(i as isize) - 1, (i as isize) + 1] {
                if (ki >= 0) && (ki < h as isize) && !mask[(ki as usize, j)] {
                    counter += 1.0;
                    value += T::into(frame[(ki as usize, j, k)]);
                }
            }
            for kj in [(j as isize) - 1, (j as isize) + 1] {
                if (kj >= 0) && (kj < w as isize) && !mask[(i, kj as usize)] {
                    counter += 1.0;
                    value += T::into(frame[(i, kj as usize, k)]);
                }
            }

            let value = if dither {
                (fastrand::f32() < (value / counter)) as u32 as f32
            } else {
                value / counter
            };

            *out_val = T::clamp(value);
        }
    });

    let output = if needs_squeeze {
        output.into_dyn().squeeze()
    } else {
        output.into_dyn()
    };
    Ok(output)
}

// ------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use ndarray::{array, ArrayD};

    use crate::transforms::{pack_single, unpack_single};

    #[test]
    fn unpack_pack_axis0() {
        let packed: ArrayD<u8> = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]].into_dyn();
        let unpacked = unpack_single(&packed.view(), 0).unwrap();
        let repacked = pack_single(&unpacked.view(), 0).unwrap();
        assert_eq!(packed.into_dyn(), repacked);
    }

    #[test]
    fn unpack_pack_axis1() {
        let packed: ArrayD<u8> = array![[1, 2, 3], [4, 5, 6], [7, 8, 9]].into_dyn();
        let unpacked = unpack_single(&packed.view(), 1).unwrap();
        let repacked = pack_single(&unpacked.view(), 1).unwrap();
        assert_eq!(packed, repacked);
    }

    #[test]
    fn unpack_pack_axis0_channels() {
        let packed: ArrayD<u8> = array![
            [[1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6]],
            [[7, 7], [8, 8], [9, 9]]
        ]
        .into_dyn();
        let unpacked = unpack_single(&packed.view(), 0).unwrap();
        let repacked = pack_single(&unpacked.view(), 0).unwrap();
        assert_eq!(packed.into_dyn(), repacked);
    }

    #[test]
    fn unpack_pack_axis1_channels() {
        let packed: ArrayD<u8> = array![
            [[1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6]],
            [[7, 7], [8, 8], [9, 9]]
        ]
        .into_dyn();
        let unpacked = unpack_single(&packed.view(), 1).unwrap();
        let repacked = pack_single(&unpacked.view(), 1).unwrap();
        assert_eq!(packed.into_dyn(), repacked);
    }
}
