pub mod node;
pub mod tensor;
pub mod function;

use ndarray;

use num_traits;
use num;

use std::fmt;

pub trait Float:
    num_traits::Float
    + num_traits::NumAssignOps
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + Sized
    + 'static
{}

impl<T> Float for T where
    T: num::Float
        + num_traits::NumAssignOps
        + Copy
        + Send
        + Sync
        + fmt::Display
        + fmt::Debug
        + Sized
        + 'static
{}

/// alias for `ndarray::Array<T, IxDyn>`
pub type NdArray<T> = ndarray::Array<T, ndarray::IxDyn>;

/// alias for `ndarray::ArrayView<T, IxDyn>`
pub type NdArrayView<'a, T> = ndarray::ArrayView<'a, T, ndarray::IxDyn>;

/// alias for `ndarray::ArrayViewMut<T, IxDyn>`
pub type NdArrayViewMut<'a, T> = ndarray::ArrayViewMut<'a, T, ndarray::IxDyn>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
