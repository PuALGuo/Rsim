nvmain中仍需修改的地方

//1. 时序修正   
//--ritecycle比预想的早了一个周期
2. 正确的数据 
--论文中的ReRAM读写矩阵时间
3. 数据reuse方案
//--地址更改
//--X方向滑动
--Y方向滑动

已经修正的地方

1. nextIssuable设计                                             
--交叉时序
2. compute内部（linear）        								             
3. compute内部（loop）
4. load外部（loop）
5. compute外部（loop）
6. load/compute功能 

7. 时序修正