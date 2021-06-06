package com.easy;

import org.w3c.dom.ls.LSException;

import java.util.*;

public class Solution {
    public static void main(String[] args) {
        ListNode node1 = new ListNode(1);
        ListNode node2 = new ListNode(2);
        ListNode node3 = new ListNode(3);
        ListNode node4 = new ListNode(4);
        ListNode node5 = new ListNode(3);
        ListNode node6 = new ListNode(2);
        node1.next = node2;
        node2.next = node3;
        node3.next = node4;
        node4.next = node5;
        node5.next = node6;
        System.out.println(isPalindrome(
                "a--565a"));

    }
    //112. 路径总和
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null)return false;
        if (root.left == null && root.right == null)
            return targetSum - root.val == 0;
        return hasPathSum(root.left,targetSum - root.val)
                || hasPathSum(root.right,targetSum - root.val);
    }
    public int majorityElement(int[] nums) {
        //摩尔投票找出出现最多的那个数
        int count = 0, ans = 0;
        for (int num : nums) {
            if (count == 0) {
                ans = num;
                count = 1;
            } else if (ans == num) {
                count++;
            } else count--;
        }
        //出现最多的数不一定满足大于数组一半的要求
        if (count >= 2){
            return ans;
        }else {
            count = 0;
            for (int num : nums) {
                if (num == ans)
                    count++;
                else count--;
            }
        }
        return count > 0 ? ans : -1;
    }
    //1331. 数组序号转换
    public int[] arrayRankTransform(int[] arr) {
        int[] copy = Arrays.copyOf(arr, arr.length);
        Arrays.sort(copy);
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0,j = 1; i < copy.length; i++) {
            if (!map.containsKey(copy[i]))
                map.put(copy[i], j);
        }
        for (int i = 0; i < arr.length; i++) {
            arr[i] = map.get(arr[i]);
        }
        return arr;
    }
    //1544. 整理字符串
    public String makeGood(String s) {
        StringBuilder sb = new StringBuilder(s);
        int len = 0;
        while (len != sb.length()){
            len = sb.length();
            for (int i = 0; i < sb.length() - 1; i++) {
                if (Math.abs(sb.charAt(i) - sb.charAt(i+1)) == 32){
                    sb.delete(i,i+2);
                    break;
                }
            }
        }
        return sb.toString();
    }
    //125. 验证回文串
    public static boolean isPalindrome(String s) {
        char[] array = s.toUpperCase().toCharArray();
        int pre = 0, tail = array.length - 1;
        while (pre < tail) {
            while (pre < tail && ((array[pre] < 'A' || array[pre] > 'Z') && (array[pre] < '0' || array[pre] > '9'))) {
                pre++;
            }
            while (tail > pre && ((array[tail] < 'A' || array[tail] > 'Z') && (array[tail] < '0' || array[tail] > '9'))) {
                tail--;
            }
            if (array[pre] == array[tail]) {
                pre++;
                tail--;
            } else return false;
        }
        return true;
    }
    //225. 用队列实现栈
    class MyStack {
        private Deque deque = new ArrayDeque<String>();
        /** Initialize your data structure here. */
        public MyStack() {

        }

        /** Push element x onto stack. */
        public void push(int x) {
            deque.push(x);
        }

        /** Removes the element on top of the stack and returns that element. */
        public int pop() {
            return (int) deque.removeLast();
        }

        /** Get the top element. */
        public int top() {
            return (int) deque.peekFirst();
        }

        /** Returns whether the stack is empty. */
        public boolean empty() {
            return deque.isEmpty();
        }
    }
    // 面试题 02.06. 回文链表
    public static boolean isPalindrome(ListNode head) {

        if(head == null)return true;
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode newHead = null;
        ListNode tmp = slow.next;
        while (tmp != null){
            ListNode buff = tmp.next;
            tmp.next = newHead;
            newHead = tmp;
            tmp = buff;
        }
        while (newHead != null && head != null){
            if (newHead.val == head.val){
                newHead = newHead.next;
                head = head.next;
            }else return false;
        }
        return true;
    }
    //67. 二进制求和
    public static String addBinary(String a, String b) {
        int yu = 0;
        StringBuilder sb = new StringBuilder();
        //chang
       if(a.length() < b.length() ){
           String tmp = a;
           a = b;
           b = tmp;
       }
        int i = b.length() - 1;
        for(; i >= 0;i--){
            int tmp = Integer.parseInt(String.valueOf(b.charAt(i))) + Integer.parseInt(String.valueOf(a.charAt(i))) + yu;
            yu = tmp / 2;
            sb.append(tmp % 2);
        }
        for(int j = a.length() - b.length();j>=0;j--){
           yu = Integer.parseInt(String.valueOf(a.charAt(j))) + yu /2 ;
           sb.append(yu == 1 ? yu : Integer.parseInt(String.valueOf(b.charAt(j))) + yu);
        }
        return sb.reverse().toString();
    }
    //1773. 统计匹配检索规则的物品数量
    public int countMatches(List<List<String>> items, String ruleKey, String ruleValue) {
        int rs = 0;
        for (List<String> item : items) {
            if (ruleKey.equals("type") && item.get(0).equals(ruleValue)
            ||ruleKey.equals("color") && item.get(1).equals(ruleValue)
            ||ruleKey.equals("name") && item.get(2).equals(ruleValue))
                rs++;
        }
        return rs;
    }
    //1281. 整数的各位积和之差
    public int subtractProductAndSum(int n) {
        int ji = 1,he = 0;
        while(n != 0){
            ji *= n % 10;
            he += n % 10;
            n = n / 10;
        }
        return ji - he;
    }

    public static int[] createTargetArray(int[] nums, int[] index) {
        StringBuilder sb = new StringBuilder();
        int ind = 0,num = 0;
        for(int i = 0;i < nums.length;i++){
            ind = index[i];
            num = nums[i];
            sb.insert(ind,num);
        }
        int[] target = new int[sb.length()];
        for (int i = 0; i < sb.length(); i++) {
            target[i] = Integer.parseInt(String.valueOf(sb.charAt(i)));
        }
        return target;
    }
    //面试题 17.04. 消失的数字
    public int missingNumber(int[] nums) {
         int rs = nums.length * (nums.length + 1) / 2 ;
        for (int num : nums) {
            rs -= num;
        }
        return rs;
    }

     public static class ListNode {
      int val;
      ListNode next;
      ListNode(int x) { val = x; }
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
}
