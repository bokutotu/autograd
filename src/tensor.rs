use std::cell::UnsafeCell;
use crate::{Float, NdArray};
use ndarray::IxDyn;

// use crate::traits::{Node};

#[derive(Debug)]
pub struct Tensor<'a, T: Float> {
    pub input: UnsafeCell<NdArray<T>>,
    pub grad:  UnsafeCell<NdArray<T>>,
    pub dim: &'a [usize]
}

impl<'a, T: Float> Tensor<'a, T> {
    pub fn new(dim: &'a [usize]) -> Self {
        let array = NdArray::<T>::zeros(IxDyn(dim));
        Self {
            input: UnsafeCell::new(array.clone()),
            grad:  UnsafeCell::new(array.clone()),
            dim: dim,
        }
    }
}
