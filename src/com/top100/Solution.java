package com.top100;

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





      public static class ListNode {
      int val;
      ListNode next;
      ListNode() {}
      ListNode(int val) { this.val = val; }
      ListNode(int val, ListNode next) { this.val = val; this.next = next; }
  }
}
