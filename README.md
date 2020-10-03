# achieve-anchor-box

## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0+](http://pytorch.org)

#### 以每个像素为中心，生成多个大小和宽高比不同的锚框。
#### 交并比是两个边界框相交面积与相并面积之比。
#### 在训练集中，为每个锚框标注两类标签：一是锚框所含目标的类别；二是真实边界框相对锚框的偏移量。
#### 预测时，可以使用非极大值抑制来移除相似的预测边界框，从而令结果简洁。
