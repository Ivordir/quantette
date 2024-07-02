//! Contains various traits needed across the crate.

use num_traits::{AsPrimitive, NumAssignOps, NumOps, Zero};
use palette::cast::ArrayCast;
use std::alloc;

/// Types that may be cast to and from a fixed sized array.
///
/// Quantization functions in `quantette` operate over a color type/space.
/// These types must implement [`ArrayCast`] where `Component` is the data type and `N` is the
/// number of channels/dimensions.
pub trait ColorComponents<Component, const N: usize>:
    ArrayCast<Array = [Component; N]> + Copy + 'static
{
}

impl<Color, Component, const N: usize> ColorComponents<Component, N> for Color where
    Color: ArrayCast<Array = [Component; N]> + Copy + 'static
{
}

/// A numerical trait used to prevent overflow when summing any `Count` number of `Self`s.
///
/// That is, the `Sum` type needs to be able to represent the value `Count::MAX * Self::MAX` without overflowing.
/// For floating point types, this constraint is relaxed,
/// and the `Sum` type simply needs to have more precision (if possible).
pub trait SumPromotion<Count>: Copy + Into<Self::Sum> + 'static
where
    Count: Into<Self::Sum>,
{
    /// The promotion type used for the sum.
    ///
    /// For integral types, this needs to be able to represent the value
    /// `Count::MAX * Self::MAX` without overflowing.
    type Sum: Zero + NumOps + NumAssignOps + AsPrimitive<Self>;
}

impl SumPromotion<u32> for u8 {
    type Sum = u64;
}

impl SumPromotion<u32> for u16 {
    type Sum = u64;
}

impl SumPromotion<u32> for f32 {
    type Sum = f64;
}

impl SumPromotion<u32> for f64 {
    type Sum = f64;
}

/// A marker trait signifying that the "zero value" for the type can be represented by
/// the all zeros bit pattern.
///
/// This trait is necessary to be able to allocate large, multi-dimensional, fixed sized arrays directly on the heap.
/// [`Box::default`] cannot be used, since [`Default`] is only implemented for fixed sized arrays up to length 32.
/// Additionally, while the compiler may allocate fixed sized arrays with a single dimension
/// directly on the heap, it seems to not do the same for multi-dimensional, fixed sized arrays.
/// Without this trait, this could potentially cause a stack overflow.
///
/// # Safety
/// The zero value of the type must be representable by the all zeros bit pattern.
#[allow(unsafe_code)]
pub unsafe trait ZeroedIsZero: Sized + Copy {
    /// Allocates the "zero value" of the type directly on the heap by zeroing memory.
    #[must_use]
    fn box_zeroed() -> Box<Self> {
        unsafe {
            let layout = alloc::Layout::new::<Self>();
            let ptr = alloc::alloc_zeroed(layout).cast::<Self>();
            if ptr.is_null() {
                alloc::handle_alloc_error(layout)
            }
            Box::from_raw(ptr)
        }
    }

    /// Sets `self` to the "zero value" by zeroing memory.
    fn fill_zero(&mut self) {
        *self = unsafe { std::mem::zeroed() };
    }
}

#[allow(unsafe_code)]
unsafe impl ZeroedIsZero for u8 {}

#[allow(unsafe_code)]
unsafe impl ZeroedIsZero for u16 {}

#[allow(unsafe_code)]
unsafe impl ZeroedIsZero for u32 {}

#[allow(unsafe_code)]
unsafe impl ZeroedIsZero for u64 {}

#[allow(unsafe_code)]
unsafe impl ZeroedIsZero for i8 {}

#[allow(unsafe_code)]
unsafe impl ZeroedIsZero for i16 {}

#[allow(unsafe_code)]
unsafe impl ZeroedIsZero for i32 {}

#[allow(unsafe_code)]
unsafe impl ZeroedIsZero for i64 {}

#[allow(unsafe_code)]
unsafe impl ZeroedIsZero for f32 {}

#[allow(unsafe_code)]
unsafe impl ZeroedIsZero for f64 {}

#[allow(unsafe_code)]
unsafe impl<T: ZeroedIsZero, const N: usize> ZeroedIsZero for [T; N] {}
