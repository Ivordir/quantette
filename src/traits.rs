use std::{alloc, iter::Sum};

use num_traits::{AsPrimitive, NumAssignOps, NumOps, Zero};
use palette::cast::ArrayCast;

pub trait ColorComponents<Component, const N: usize>:
    ArrayCast<Array = [Component; N]> + Copy + 'static
{
}

impl<Color, Component, const N: usize> ColorComponents<Component, N> for Color where
    Color: ArrayCast<Array = [Component; N]> + Copy + 'static
{
}

pub trait SumPromotion<Count>: Zero + Copy + Into<Self::Sum> + 'static
where
    Count: Into<Self::Sum>,
{
    type Sum: Zero + Sum + NumOps + NumAssignOps + AsPrimitive<Self>;
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

///
/// # Safety
/// The zero value of the type must be representable by the all zeros bit pattern
#[allow(unsafe_code)]
pub unsafe trait ZeroedIsZero: Sized + Copy {
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
