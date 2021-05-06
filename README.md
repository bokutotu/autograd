# Autograd
このクレートはRustで自動微分を実装したクレートです。
簡単に自動微分を実装できます。

# Example

```rust
use autograd::function::*;
use autograd::node::Node;
use autograd::tensor::Tensor;

fn main() {
    // 勾配情報を持つtensorの定義(この場合はスカラー)
    let x = Tensor::new(&[]);

    // y = x * x + xの計算グラフを定義する
    let y = add(&product(&x, &x), &x);

    // 計算グラフの購買情報をリセット
    y.zero_grad();
    
    // 購買情報を伝搬する前にグラフの最も下のノードの勾配をセットする
    y.set_grad();
    
    // 計算グラフの計算を行う
    y.forward();

    // 計算グラフの勾配を計算する
    y. backward();
}
```