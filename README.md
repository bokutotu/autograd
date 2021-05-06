# Autograd
このクレートはRustで自動微分を実装したクレートです。
簡単に自動微分を実装できます。

# Example

```rust
use autograd::function::*;
use autograd::node::Node;
use autograd::tensor::Tensor;

// 10,10の勾配情報を持つtensorの定義
let x = Tensor::new(&[10,10]);
let y = Tensor::new(&[10,10]);

// x + yの勾配情報を持つ計算グラフの定義
let z = add(&x, &y);

// 計算グラフの計算を行う
z.forward();

// 計算グラフの勾配情報をリセット
z.zerograd();

// 計算グラフにおいて勾配を計算
z.backward();
```