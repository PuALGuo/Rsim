nvmain中仍需修改的地方

//1. 正确的数据 
//--论文中的ReRAM读写矩阵时间

bug列表
//1. 一连串load后接compute的问题
//2. 时钟隔离
//3. Func_n阻塞
4. X滑动和Y滑动的差异性
--关于数据复用


已经修正的地方

1. nextIssuable设计                                             
--交叉时序
2. compute内部（linear）        								             
3. compute内部（loop）
4. load外部（loop）
5. compute外部（loop）
6. load/compute功能 

7. 时序修正
8. reuse方案
--内部的滑动设计（X，Y）


补丁说明（以前的就算了）
*-------------------------------------------*
2019.5.10
1. 时钟隔离，drain
2. 将load，compute改为一个队列，防止出现load总是抢占compute的问题
3. Func_n阻塞
--将一些公有变量变成独立的私有变量
4. 添加了Transfer()，旨在实现外部搬移到ReRAM上的时序消耗
5. 添加了真实物理数据