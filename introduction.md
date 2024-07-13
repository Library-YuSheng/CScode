# Markdown使用手册

## 基本语法

1. **标题**
   #一级标题
   #二级标题
   ...
2. **引用**
   > 原来这种效果是引用啊！
3. **列表**
   1. 无序列表
      - 列表1
      + 列表2（不推荐）
      * 列表3（不推荐）
   2. 有序列表
   3. 嵌套
   4. TodoList
      - [x] a
      - [ ] b
      - [ ] c
4. **表格**
   | 左对齐 | 居中对齐 | 右对齐|  
   | :----- | :----: | ----: |  
   | a | b | c |
5. **段落**
   - 换行 —— 两个以上空格后回车/空一行  
   - 分割线 —— 三个*
     ***
   - 字体
     | 字体 | 代码 |
     |:--:|:--:|
     |*斜体*|* *|
     |==高亮==|== ==|
     |**粗体**|** **|
     |***粗斜体***|*** ***|
     |~~删除~~|~~ ~~|
     |<u>下划线</u>|`<u> </u>`|
   - 脚注
     本篇说明的参考文献[^1]！
6. **代码**

   ```c++
   #include<iostream>
   using namespace std;
   int main(){
    print("hello world");
   }
   ```

   `print("hello world);`
7. **超链接**
   - 更多使用教程可参考[网站](https://www.runoob.com/markdown/md-link.html)
   - 查看更多使用功能请[点击连接][教程]
8. **图片**
   - 使用图床保存图片并实现插入
   [路过图床](https://imgse.com/)
   - 使用markdown语法插入
   [![pFZHwAe.jpg](https://s11.ax1x.com/2024/01/23/pFZHwAe.jpg)](https://imgse.com/i/pFZHwAe)
   - 使用html语言实现调整图片大小和位置功能
   <a href="https://imgse.com/i/pFZHwAe">
   <div align=center><img src="https://s11.ax1x.com/2024/01/23/pFZHwAe.jpg" alt="pFZHwAe.jpg" border="0" width="80%" height="60%"/></div></a> 

## 其他操作

- **插入latex公式**
  - 行内显示：$f(x)=ax+b$
  - 块内显示：  
  $$
  \begin{Bmatrix}
  a & b  \\
  c & d
  \end{Bmatrix}
  $$
- **html/css语法**
  - ctrl+shift+p 搜索 "Markdown Preview Enhanced:Customize CSS" 在style中使用css语法更改标题格式等
- **个性化设置**
File-Preferences-Settings

## 导出为PDF文档

- Open in Browser——手动另存为PDF文档

[教程]: https://www.runoob.com/markdown/md-link.html
[^1]: https://www.bilibili.com/video/BV1bK4y1i7BY/?spm_id_from=333.337.search-card.all.click&vd_source=f704381cd2edf4eea836e5e635d8e9ba