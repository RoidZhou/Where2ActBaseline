# Where2Act

## 收集训练数据
### 离线随机数据采样
1) 先选交互点，再选交互方向
首先随机抽取一个位置p在ground-truth铰接部件上进行交互。然后，我们从p周围切平面上方的半球随机抽取一个交互方向R∈SO(3)，始终朝向正法线方向，并尝试查询交互结果参数化为（p， R） 