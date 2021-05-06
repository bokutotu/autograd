use crate::tensor::Tensor;
use crate::{Float};
use crate::node::Node;

macro_rules! impl_struct_and_fn {
    ($func_name: ident, $struct_name: ident) => {
        pub fn $func_name<'a, F: Float>(x: &'a (dyn Node<'a, F> + 'a), y: &'a (dyn Node<'a, F> + 'a))
        -> $struct_name<'a, F> 
        {
            let z = Tensor::<'a, F>::new(x.shape());
            $struct_name {
                x: x,
                y: y,
                z: z,
            }
        }

        pub struct $struct_name<'a, F: Float> {
            pub x: &'a dyn Node<'a, F>,
            pub y: &'a dyn Node<'a, F>,
            pub z: Tensor<'a, F>,
        }
    }
}

impl_struct_and_fn!(add, AddNode);
impl_struct_and_fn!(product, ProductNode);
// impl_struct_and_fn!(matmul, MatmulNode);

pub fn matmul<'a, F: Float>(x: &'a (dyn Node<'a, F> + 'a), y: &'a (dyn Node<'a, F> + 'a), dim: &'a [usize;2]) 
-> MatmulNode<'a, F> 
{
    let z = Tensor::<'a, F>::new(dim);
    MatmulNode {
        x: x,
        y: y,
        z: z,
    }
}

pub struct MatmulNode<'a, F:Float> {
    pub x: &'a dyn Node<'a, F>,
    pub y: &'a dyn Node<'a, F>,
    pub z: Tensor<'a, F>,
}

// pub fn add<'a, F: Float>(x: &'a (dyn Node<'a, F> + 'a), y: &'a (dyn Node<'a, F> + 'a)) 
// -> AddNode<'a, F>
// {
//     let z = Tensor::<'a, F>::new(x.shape());
//     AddNode {
//         x: x,
//         y: y,
//         z: z,
//     }
// }

// pub struct AddNode<'a, F: Float> {
//     pub x: &'a dyn Node<'a, F>,
//     pub y: &'a dyn Node<'a, F>,
//     pub z: Tensor<'a, F>,
// }

// pub fn product<'a, F: Float>(x: &'a (dyn Node<'a, F> + 'a), y: &'a (dyn Node<'a, F> + 'a))
// -> ProductNode<'a, F> 
// {
//     let z = Tensor::<'a, F>::new(x.shape());
//     ProductNode{
//         x: x,
//         y: y,
//         z: z,
//     }
// }

// pub struct ProductNode<'a, F: Float> {
//     pub x: &'a dyn Node<'a, F>,
//     pub y: &'a dyn Node<'a, F>,
//     pub z: Tensor<'a, F>,
// }
