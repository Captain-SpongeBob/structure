package com.app;

public class Main {
    {
        System.out.println("------");
    }
    public static void main(String[] args) {
        Main main = new Main();
        Main main1 = new Main();
        Main main2= new Main();
        Main main3 = new Main();
    }
}
/**
 假设有一个矩阵，矩阵由 0 和 1 数字组成。其中1代表这个节点可达，0代表这个节点不可达，从左上角第一个节点出发到右下角最后一个节点，只能上下左右移动，初始节点数为 1，移动一步节点数加 1，问最少需要经过多少节点可以达到。

 请实现计算最短路径的函数。如果不可达返回0。
 **/