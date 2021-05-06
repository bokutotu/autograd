use autograd::{function::*, node::Node, tensor::Tensor};

#[test]
fn three_elem_test() {
    let x = Tensor::<f32>::new(&[]);
    unsafe {
        *x.get_input() += 3.;
    }
    
    // y = x * x + x
    let tmp = product(&x, &x);
    let y = add(&tmp, &x);

    y.zero_grad();
    y.set_grad();
    y.forward();
    y.backward();

    unsafe {
        assert_eq!([12.], (*y.get_input()).as_slice().unwrap()); 
        assert_eq!([7.], (*x.get_grad()).as_slice().unwrap());
    }
}
