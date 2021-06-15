package com.top100;

import java.util.Arrays;
import java.util.HashMap;

public class Solution {
    public static void main(String[] args) {
        ListNode node1 = new ListNode(4);
        ListNode node2 = new ListNode(2);
        ListNode node3 = new ListNode(1);
        ListNode node4 = new ListNode(3);
        node1.next = node2;
        node2.next = node3;
        node3.next = node4;
        sortList(node1);
    }

    public static ListNode sortList(ListNode head) {
       return split(head);
    }
    //
    HashMap<TreeNode,Integer> f = new HashMap<>();
    HashMap<TreeNode,Integer> g = new HashMap<>();
    public int rob(TreeNode root) {
        dfs(root);
        return Math.max(f.getOrDefault(root, 0), g.getOrDefault(root, 0));
    }
    public void dfs(TreeNode root){
        if (root == null)return;
        dfs(root.left);
        dfs(root.right);
        f.put(root,root.val + g.getOrDefault(root.left,0) + g.getOrDefault(root.right,0));
        g.put(root,Math.max(f.getOrDefault(root.left,0),g.getOrDefault(root.left,0))
                +Math.max (f.getOrDefault(root.right,0),g.getOrDefault(root.right,0)));
    }
    //312. 戳气球
    int[][] rec;
    int[] val;
    public int maxCoins(int[] nums) {
        val = new int[nums.length + 2];
        val[0] = 1;
        val[nums.length + 1] = 1;
        for (int i = 1; i < val.length - 1; i++) {
            val[i] = nums[i - 1];
        }
        rec = new int[val.length][val.length];
        for (int[] ints : rec) {
            Arrays.fill(ints,-1);
        }
        return solve(0,val.length - 1);
    }
    public int solve(int left,int right){
        if (left >= right - 1)return 0;
        if (rec[left][right] != -1)return rec[left][right];
        for (int i = left + 1; i < right; i++) {
            int sum = val[left] * val[i] * val[right];
            sum += solve(left,i) + solve(i,right);
            rec[left][right] = Math.max(rec[left][right],sum);
        }
        return rec[left][right];
    }
    //148. 排序链表
    public static ListNode split(ListNode head){
        //只有一个节点的时候跳，否则进入死循环
        if (head == null || head.next == null)return head;
        ListNode slow = head,fast = head;
        ListNode node = head;
        while (fast != null && fast.next != null){
            node = slow;
            slow = slow.next;
            fast = slow.next;
        }
        node.next = null;
        ListNode l = split(head);
        ListNode r = split(slow);
        return merge(l,r);
    }
    public static ListNode merge(ListNode l, ListNode r){
        //临时头节点
        ListNode tmp = new ListNode();
        ListNode pre = tmp;
        while (l != null && r != null){
            if (l.val < r.val){
                pre.next = l;
                pre = pre.next;
                l = l.next;
            }else {
                pre.next = r;
                pre = pre.next;;
                r = r.next;
            }
        }
        pre.next = l == null ? r : l;
        return tmp.next;
    }




  public class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode() {}
      TreeNode(int val) { this.val = val; }
      TreeNode(int val, TreeNode left, TreeNode right) {
          this.val = val;
          this.left = left;
          this.right = right;
      }
  }
  public static class ListNode {
      int val;
      ListNode next;
      ListNode() {}
      ListNode(int val) { this.val = val; }
      ListNode(int val, ListNode next) { this.val = val; this.next = next; }
  }
}
