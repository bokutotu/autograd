use crate::{Float, NdArray};
use crate::tensor::Tensor;
use crate::function::{
    AddNode, 
    ProductNode,
    MatmulNode,
};

use ndarray::{IxDyn, linalg::general_mat_mul, Ix2};

pub trait Node<'a, F: Float> {
    fn get_input(&self) -> *mut NdArray<F>;
    fn get_grad(&self) -> *mut NdArray<F>;
    fn reset_grad(&self);
    fn shape(&self) -> &'a [usize];
    fn forward(&self) ;
    fn backward(&self);
    fn zero_grad(&self);
    fn set_grad(&self);
}

impl<'a, F: Float> Node<'a, F> for Tensor<'a, F> {

    fn get_input(&self) -> *mut NdArray<F> {
        self.input.get()
    }

    fn get_grad(&self) -> *mut NdArray<F> {
        self.grad.get()
    }

    fn reset_grad(&self) {
        let borrow = self.grad.get();
        unsafe {
            *borrow = NdArray::<F>::zeros(IxDyn(self.dim));
        }
    }

    fn shape(&self) -> &'a [usize] {
        self.dim
    }

    fn forward(&self) {}

    fn backward(&self) {}

    fn zero_grad(&self) {
        self.reset_grad();
    }

    fn set_grad(&self) {
        unsafe {
            *self.get_grad() = NdArray::<F>::ones(self.dim);
        }
    }

}

impl<'a, F: Float> Node<'a, F> for AddNode<'a, F> {

    fn get_input(&self) -> *mut NdArray<F> {
        self.z.input.get()
    }

    fn get_grad(&self) -> *mut NdArray<F> {
        self.z.grad.get()
    }

    fn reset_grad(&self) {
        let borrow = self.z.grad.get();
        unsafe {
            *borrow = NdArray::<F>::zeros(IxDyn(self.z.dim));
        }
    }

    fn shape(&self) -> &'a [usize] {
        self.z.dim
    }

    fn forward(&self) {
        self.x.forward();
        self.y.forward();
        let borrow_x = self.x.get_input();
        let borrow_y = self.y.get_input();
        let borrow_z = self.z.get_input();
        unsafe {
            *borrow_z = (*borrow_x).clone() + (*borrow_y).clone();
        }
    }

    fn backward(&self) {
        let borrow_x = self.x.get_grad();
        let borrow_y = self.y.get_grad();
        let borrow_z = self.z.get_grad();
        unsafe {
            *borrow_x += &(*borrow_z).clone();
            *borrow_y += &(*borrow_z).clone();
        }
        self.x.backward();
        self.y.backward();
    }

    fn zero_grad(&self) {
        self.x.reset_grad();
        self.y.reset_grad();
        self.z.reset_grad();
    }

    fn set_grad(&self) {
        unsafe {
            *self.z.get_grad() = NdArray::<F>::ones(self.z.shape());
        }
    }
}

impl<'a, F: Float> Node<'a, F> for ProductNode<'a, F> {

    fn get_input(&self) -> *mut NdArray<F> {
        self.z.input.get()
    }

    fn get_grad(&self) -> *mut NdArray<F> {
        self.z.grad.get()
    }

    fn reset_grad(&self) {
        let borrow = self.z.grad.get();
        unsafe {
            *borrow = NdArray::<F>::zeros(IxDyn(self.z.dim));
        }
    }

    fn shape(&self) -> &'a [usize] {
        self.z.dim
    }

    fn forward(&self) {
        self.x.forward();
        self.y.forward();
        let borrow_x = self.x.get_input();
        let borrow_y = self.y.get_input();
        let borrow_z = self.z.get_input();
        unsafe {
            *borrow_z = (*borrow_x).clone() * (*borrow_y).clone();
        }
    }

    fn backward(&self) {
        let borrow_x_input = self.x.get_input();
        let borrow_y_input = self.y.get_input();
        let borrow_z_grad = self.z.get_grad();
        let borrow_x_grad = self.x.get_grad();
        let borrow_y_grad = self.y.get_grad();
        unsafe {
            *borrow_x_grad += &((*borrow_z_grad).clone() * (*borrow_y_input).clone());
            *borrow_y_grad += &((*borrow_z_grad).clone() * (*borrow_x_input).clone());
        }
        self.x.backward();
        self.y.backward();
    }

    fn zero_grad(&self) {
        self.x.reset_grad();
        self.y.reset_grad();
        self.z.reset_grad();
    }

    fn set_grad(&self) {
        unsafe {
            *self.z.get_grad() = NdArray::<F>::ones(self.z.shape());
        }
    }
}

impl<'a> Node<'a, f32> for MatmulNode<'a, f32> {

    fn get_input(&self) -> *mut NdArray<f32> {
        self.z.input.get()
    }

    fn get_grad(&self) -> *mut NdArray<f32> {
        self.z.grad.get()
    }

    fn reset_grad(&self) {
        let borrow = self.z.grad.get();
        unsafe {
            *borrow = NdArray::<f32>::zeros(IxDyn(self.z.dim));
        }
    }

    fn shape(&self) -> &'a [usize] {
        self.z.dim
    }

    fn forward(&self) {
        self.x.forward();
        self.y.forward();
        let borrow_x = self.x.get_input();
        let borrow_y = self.y.get_input();
        let borrow_z = self.z.get_input();
        unsafe {
            let x = (*borrow_x).clone().into_dimensionality::<Ix2>().unwrap();
            let y = (*borrow_y).clone().into_dimensionality::<Ix2>().unwrap();
            let mut z = (*borrow_z).clone().into_dimensionality::<Ix2>().unwrap();
            general_mat_mul(1., &x, &y, 0., &mut z);
            *self.z.get_input() += &z;
        }
    }

    fn backward(&self) {
        unsafe {
            let x_input = (*self.x.get_input()).clone().into_dimensionality::<Ix2>().unwrap();
            let y_input = (*self.y.get_input()).clone().into_dimensionality::<Ix2>().unwrap();

            let mut x_grad = (*self.x.get_grad()).clone().into_dimensionality::<Ix2>().unwrap();
            let mut y_grad = (*self.y.get_grad()).clone().into_dimensionality::<Ix2>().unwrap();
            let z_grad = (*self.z.get_grad()).clone().into_dimensionality::<Ix2>().unwrap();
            
            general_mat_mul(1., &z_grad, &y_input.reversed_axes(), 0., &mut x_grad);
            general_mat_mul(1., &x_input.reversed_axes(), &z_grad, 0., &mut y_grad);

            *self.x.get_grad() += &x_grad;
            *self.y.get_grad() += &y_grad;
        }
        self.x.backward();
        self.y.backward();
    }

    fn zero_grad(&self) {
        self.x.reset_grad();
        self.y.reset_grad();
        self.z.reset_grad();
    }

    fn set_grad(&self) {
        unsafe {
            *self.z.get_grad() = NdArray::<f32>::ones(self.z.shape());
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::function::{add, product, matmul};
    use crate::tensor::Tensor;
    use crate::node::{Node};
    use crate::NdArray;
    use ndarray::{IxDyn, array};

    #[test]
    fn add_test() {
        let x = Tensor::<f32>::new(&[10]);
        let y = Tensor::<f32>::new(&[10]);
        unsafe {
            *x.get_input() = NdArray::<f32>::ones(IxDyn(&[10])) * 10f32;
            *y.get_input() = NdArray::<f32>::ones(IxDyn(&[10])) * 5f32;
        }
        let z = add(&x, &y);
        z.forward();
        z.set_grad();
        z.backward();
        let borrow_input = z.get_input();
        let borrow_x_grad = x.get_grad();
        unsafe {assert_eq!((*borrow_input).as_slice().unwrap(), [15f32; 10]);}
        unsafe {assert_eq!((*borrow_x_grad).as_slice().unwrap(), [1f32; 10]);}
    }

    #[test]
    fn product_test() {
        let x = Tensor::<f32>::new(&[10]);
        let y = Tensor::<f32>::new(&[10]);
        let z = product(&x, &y);

        unsafe {
            *x.get_input() = NdArray::<f32>::ones(IxDyn(&[10])) * 2f32;
            *y.get_input() = NdArray::<f32>::ones(IxDyn(&[10])) * 5f32;
        }

        z.forward();
        z.set_grad();
        z.backward();

        let borrow_input = z.get_input();
        let borrow_x_grad = x.get_grad();
        unsafe {assert_eq!((*borrow_input).as_slice().unwrap(), [10f32; 10]);}
        unsafe {assert_eq!((*borrow_x_grad).as_slice().unwrap(), [5f32; 10]);}
    }

    #[test]
    fn matmul_test() {
        let x = Tensor::<f32>::new(&[1,3]);
        let y = Tensor::<f32>::new(&[3, 5]);
        let z = matmul(&x, &y, &[1,5]);

        unsafe {
            (*x.get_input()) += &array![[1., 2., 3.]];
            (*y.get_input()) += &array![[1., 2., 3., 4., 5.],
                                        [1., 2., 3., 4., 5.],
                                        [1., 2., 3., 4., 5.]];
        }

        z.forward();
        z.set_grad();
        z.backward();
        unsafe { println!("{:?}", *z.get_input()); }
        unsafe { println!("{:?}", *x.get_grad()); }
    }
}
