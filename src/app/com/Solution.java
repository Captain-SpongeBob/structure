package app.com;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
 


public class Solution {

    public static void main(String[] args) {
        ArrayList<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(2);
        list.add(3);
        list.add(4);
        ArrayDeque<Integer> deque = new ArrayDeque<>();
        deque.add(1);
        deque.add(2);
        deque.add(3);
        deque.add(4);
    /*    Scanner in = new Scanner(System.in);
        String[] s = in.nextLine().split(" ");
        int n = Integer.parseInt(s[0]);
        int m = Integer.parseInt(s[1]);
*/
/**
        ListNode third1 = new Solution().new ListNode(1);
        ListNode third_1 = new Solution().new ListNode(2);
        ListNode third2 = new Solution().new ListNode(4);

        ListNode third3 = new Solution().new ListNode(1);
        ListNode third4 = new Solution().new ListNode(3);
        ListNode third5 = new Solution().new ListNode(4);
        ListNode third6 = new Solution().new ListNode(7);
        ListNode third7 = new Solution().new ListNode(8);

        third1.next = third_1;
        third_1.next = third2;
        third2.next = null;
        third3.next = third4;
        third4.next = third2;

        third5.next = null;
        third6.next = third7;
        third7.next = null;
        ListNode head = new Solution().new ListNode(1);
        ListNode a1 = null,a2 = null;

        TreeNode root = new TreeNode(3);
        TreeNode node1 = new TreeNode(3);
        TreeNode node2 = new TreeNode(3);
        TreeNode node5 = new TreeNode(3);
        TreeNode node6 = new TreeNode(3);
        root.left = node1;
        root.right = node2;
        node1.left = null;
        node1.right = null;
        node2.left = node5;
        node2.right = node6;
        node5.left = null;
        node5.right = null;
        node6.left = null;
        node6.right = null;
        var arr = new int[]{4,1,2,1,2};

        String p = "abba";
        String[] s = {"a","b", "ap", "ba", "app", "ban", "appl", "bana", "apply", "banan", "banana"};
        int[] nums = new int[]{1,2,3};
        int[] nums2 = new int[]{1,2,5,10,6,9,4,3};
        String s1 = "a good   example";

        System.out.println(verifyPostorder(nums2));
 **/

    }


    public static void generateParenthesisCore( List<String> ans, String tmp,int count1, int count2, int n){
        if(count1 > n || count2 > n)return;
        if(tmp.length() == n * 2){
            ans.add(tmp);
            return;
        }
        if(count2 <= count1 ){
            generateParenthesisCore(ans,tmp + "(",count1 + 1,count2,n);
            generateParenthesisCore(ans,tmp + ")",count1,count2 + 1,n);
        }
    }
    //换硬币
    public static int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];//dp每个内容表示金额为i时所需的最少金币

        Arrays.fill(dp, amount + 1);
        dp[0] = 0 ;//金额为0则所需数目为0
        for(int i = 1 ;i < amount + 1;i++){
            for(int j = 0;j < coins.length;j++){
                if(i - coins[j] > -1)
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }
    public int deepestLeavesSum1(TreeNode root) {
        if (root == null)return 0;
        LinkedList<TreeNode> deque = new LinkedList<>();
        deque.add(root);
        int max = 0;
        while (!deque.isEmpty()){
            int size = deque.size();
            max = 0;
            while (size > 0){
                size--;
                TreeNode tmp = deque.poll();
                max += tmp.val;
                if (tmp.left != null)deque.add(tmp.left);
                if (tmp.right != null)deque.add(tmp.right);
            }
        }
        return max;
    }
    public boolean existCore(char[][] board,  char[] chars ,boolean[][] visited,int row, int col,int start) {
        if (row >= board.length || col >= board[0].length)return false;
        if (start == chars.length)return true;
        boolean hasnext = false;
        if (row > -1 && row < board.length && col < board[0].length && col > -1 &&!visited[row][col] && chars[start] == board[row][col]){
            visited[row][col] = true;
            hasnext = existCore(board,chars,visited,row,col - 1,start+1)
                    || existCore(board,chars,visited,row,col + 1,start+1)
                    || existCore(board,chars,visited,row - 1,col ,start+1)
                    || existCore(board,chars,visited,row + 1,col ,start+1);
            if (!hasnext){
                visited[row][col] = false;
            }
        }

        return hasnext;
    }
    //23. 合并K个升序链表
    public ListNode mergeKLists( ListNode[] lists) {
        PriorityQueue<ListNode> queue = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val - o2.val;
            }
        });
        for (ListNode node : lists) {
            while (node != null){
                queue.add(node);
                node = node.next;
            }
        }
        ListNode rs = new ListNode();
        ListNode pre = rs;
        while (!queue.isEmpty()){
            pre.next = queue.poll();
            pre = pre.next;
        }
        return rs.next;
    }
    public  ListNode minusList(ListNode minuendList, ListNode subtrahendList) {
        ListNode m = minuendList;
        ListNode s = subtrahendList;
        String s1 = "";
        String s2 = "";
        while (m != null){
            s1 += m.val;
            m = m.next;
        }
        while (s != null){
            s2 += s.val;
            s = s.next;
        }
        int num1 = Integer.parseInt(s1);
        int num2 = Integer.parseInt(s2);
        int sub = num1 - num2;
        ListNode head = new ListNode(-1);
        if (sub == 0)return new ListNode(0);
        boolean flag = false;
        if (sub < 0)flag = true;
        while (sub != 0) {
            ListNode tmp = new ListNode(Math.abs(sub % 10));
            sub = sub / 10;
            tmp.next = head.next;
            head.next = tmp;
        }
        int tmp = head.next.val;
        if (flag)head.next.val = -tmp;
        return head.next;
    }
    public int bestCardPair (int[] cards) {
        int max = Integer.MIN_VALUE;
        for(int i = 0 ; i < cards.length;i++){
            for (int j = cards.length - 1; j > 0; j--) {
                int tmp = cards[i] + cards[j] + i - j;
                if(tmp > max )max = tmp;
            }
        }
        return max;
    }

    public int maxMoney (TreeNode root) {
        return moeDFS(root) + moeDFS2(root);
    }
    public int moeDFS(TreeNode root){
        if (root == null)return 0;
        if (root.left == null && root.right == null)
            return root.val;
        return moeDFS(root.left) + moeDFS(root.right);
    }
    public int moeDFS2(TreeNode root){
        if (root != null){
            if (root.left != null && root.left.left == null && root.left.right == null
                    || root.right != null && root.right.left == null && root.right.right == null)
                return root.val;
        }
        return 0;
    }




    //剑指 Offer 11. 旋转数组的最小数字
    public static int minArray(int[] numbers) {
        int left = 0,right = numbers.length - 1;
        int min = numbers[0];
        int index = 0;
        while(left < right){
            int mid = left + ((right - left) >> 1);
            if (numbers[mid] < numbers[right])right = mid;
            else if (numbers[mid] > numbers[right])left = mid + 1;
            else right--;
        }
        return numbers[left];
    }

    //剑指offer 最长回文字符串
    public static String longestPalindrome(String s) {
        if (s == null || s.length() == 0)return null;
        int start = 0,end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = longestPalindromeCore(i, i ,s);
            int len2 = longestPalindromeCore(i , i + 1,s);
            int len = Math.max(len1,len2);
            if (len > end - start){
                start = i - (len - 1) / 2;
                end = i + (len / 2);
            }

        }
        return s.substring(start, end + 1);
    }
    public static int longestPalindromeCore(int left, int right, String s){
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)){
            left--;
            right++;
        }
        return right - left - 1;
    }
    //剑指offer 电话簿
    public static List<String> letterCombinations(String digits) {
        if (digits == null || digits.length() == 0)return new ArrayList<>();
        HashMap<Character,String> map = new HashMap<>();
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");
        ArrayList<String> ans = new ArrayList<>();
        letterCombinationsCore(ans,"",digits,0,map);
        return ans;
    }
    public static void letterCombinationsCore(List<String> ans, String tmp, String digits, int index, HashMap<Character, String> map){
        if (tmp.length() == digits.length()){
            ans.add(tmp);
            return;
        }
        for (int i = index; i < digits.length(); i++) {
            String str = map.get(digits.charAt(i));
            for (int j = 0; j < str.length(); j++) {
                tmp += str.charAt(j);
                letterCombinationsCore(ans,tmp,digits,i + 1,map);
                tmp = tmp.substring(0, tmp.length() - 1);

            }
        }

    }
    // 34. 在排序数组中查找元素的第一个和最后一个位置
    public static int[] searchRange(int[] nums, int target) {
            if(nums.length == 0)return new int[]{-1,-1};
            int left = 0,right = nums.length - 1;
            while(left <= right){
                int mid = (left + right) / 2;
                if(nums[mid] > target)
                    right = mid - 1;
                else if(nums[mid] < target)left = mid + 1;
                else{
                    left = mid - 1;
                    right = mid + 1;
                    while(left >= 0 && nums[left] == target)left--;
                    while(right < nums.length &&nums[right] == target)right++;
                    break;
                }
            }
            if (left > right) {
                return new int[]{-1,-1};
            }
            return new int[]{left + 1, right - 1};
        }


    public int rob(int[] nums) {
        if(nums.length == 2)
            return Math.max(nums[0], nums[1]);
        int[] dp = new int[nums.length];
        for(int i = 2; i < nums.length;i++){
            dp[i] =  Math.max(nums[i] + dp[ i - 2], dp[i - 1]);
        }

        return dp[nums.length - 1];
    }
    static String uncompress(String cmpStr) {
         if (cmpStr == null || cmpStr.length() == 0)return "Error";
        char[] chars = cmpStr.toCharArray();
        for (char aChar : chars) {
            if (aChar > 'z' || aChar < 'A' )
                if (!Character.isDigit(aChar))
                    return "Error";

        }
        ArrayList<Character> characters = new ArrayList<>();
        ArrayList<Integer> nums = new ArrayList<>();
        int i = 0;
        for (; i < chars.length; ) {
            if (chars[i] <= 'z' && chars[i] >= 'A'){
                if (i + 1 < chars.length && Character.isDigit(chars[i + 1]))return "Error";
                characters.add(chars[i]);
                i++;
            }
            else if (chars[i] > 'z' || chars[i] < 'A' || !Character.isDigit(chars[i]) || chars[i] == ' ') {
                return "Error";
            }
            else {
                    StringBuilder str = new StringBuilder();
                    int j = i;
                    for (; j < chars.length && (chars[j] > 'z' || chars[j] < 'A'); j++) {
                        str.append(String.valueOf(chars[j]));
                    }
                    nums.add(Integer.parseInt(str.toString()));
                    i = j;
                }
            }
        StringBuilder sb = new StringBuilder();
        if (characters.size() == 0 )return "Error";
        for (int k = 0; k < characters.size(); k++) {
            char c = characters.get(k);
            int count = nums.get(k);
            for (int j = 0; j < count; j++) {
                sb.append(c);
            }
        }
        return sb.toString();
    }
    //347. 前 K 个高频元素
    public static int[] topKFrequent(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0 ) + 1);
        }
        PriorityQueue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }
        });
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (queue.size() == k){
                if(queue.peek()[1] < entry.getValue()){
                    queue.poll();
                    queue.add(new int[]{entry.getKey(), entry.getValue()});
                }
            }else if (queue.size() < k){
                queue.add(new int[]{entry.getKey(), entry.getValue()});
            }
        }
        int[] ans = new int[k];
        for (int an : ans) {
            an = queue.poll()[0];
        }
        return ans;
    }
    public static void sortColors(int[] nums) {
        for(int i = nums.length - 1; i >= 0 ; i-- ){
            for(int j = 0; j <= i;j++){
                if(nums[j] > nums[i]  ){
                    int tmp = nums[i];
                    nums[i] = nums[j];
                    nums[j] = tmp;
                }
            }
        }
    }

    // 647. 回文子串
    public static int countSubstrings(String s) {
        boolean[][] dp = new boolean[s.length()][s.length()];
        int ans = 0;
        for (int j = 0; j < s.length(); j++) {
            for (int i = 0; i<=j; i++) {
                if (s.charAt(i) == s.charAt(j) && (j - i < 2 || dp[i + 1][j - 1])){
                    ans++;
                    dp[i][j] = true;
                }
            }
        }
        return ans;
    }
    // 105. 从前序与中序遍历序列构造二叉树
    public TreeNode buildTree1(int[] preorder, int[] inorder) {
        if (preorder.length == 0 || inorder.length == 0 || preorder == null || inorder == null)return null;
        return buildTreeCore(preorder,0, preorder.length - 1,inorder, 0, inorder.length - 1);
    }
    public TreeNode buildTreeCore(int[] pre,int loi1,int hoi1, int[] in, int loi2,int hoi2){
        if (loi1 > hoi1 || loi2 > hoi2 )return null;
        TreeNode node = null;
        int mid = 0;
        for (int i = loi2; i <= hoi2; i++) {
            if (in[i] == pre[loi1]){
                mid = i;
                node = new TreeNode(pre[loi1]);
                break;
            }
        }
        node.left = buildTreeCore(pre, loi1 + 1,hoi1 + mid - loi2, in, loi2, mid - 1);
        node.right = buildTreeCore(pre, loi1 + mid - loi2 + 1, hoi1, in, mid + 1, hoi2);
        return node;
    }
    // 448. 找到所有数组中消失的数字
    public static List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> ans = new ArrayList<>();
        int n = nums.length;
        for (int num : nums) {
            int x = (num - 1) % n;
            nums[x] += n;
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n)ans.add(i + 1);
        }
        return ans;
    }

    // nvyouxueer
    List<List<String>> items = new ArrayList<>();
    public int countMatches(List<List<String>> items, String ruleKey, String ruleValue) {
        items.add(new ArrayList(){});
        int ans = 0;
        Map<String,Integer> map = new HashMap<>();
        map.put("type",0);
        map.put("color",1);
        map.put("name",2);
        int key = map.get(ruleKey);
        for(List<String> item : items){
            if( item.get(key) == ruleValue);
            ans++;
        }
        return ans;
    }
    public class Trie {
        private boolean is_string=false;
        private Trie next[]=new Trie[26];

        public Trie(){}

        public void insert(String word){//插入单词
            Trie root=this;
            char w[]=word.toCharArray();
            for(int i=0;i<w.length;++i){
                if(root.next[w[i]-'a']==null)root.next[w[i]-'a']=new Trie();
                root=root.next[w[i]-'a'];
            }
            root.is_string=true;
        }

        public boolean search(String word){//查找单词
            Trie root=this;
            char w[]=word.toCharArray();
            for(int i=0;i<w.length;++i){
                if(root.next[w[i]-'a']==null)return false;
                root=root.next[w[i]-'a'];
            }
            return root.is_string;
        }

        public boolean startsWith(String prefix){//查找前缀
            Trie root=this;
            char p[]=prefix.toCharArray();
            for(int i=0;i<p.length;++i){
                if(root.next[p[i]-'a']==null)return false;
                root=root.next[p[i]-'a'];
            }
            return true;
        }
    }

    // 90. 子集 II
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        dfsSubsetsWithDup(nums, ans,new ArrayList<>(),0);
        return ans;
    }
    public void dfsSubsetsWithDup(int[] nums, List<List<Integer>> ans, List<Integer> tmp ,int start){
        ans.add(new ArrayList<>(tmp));
        for (int i = start; i <nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1])continue;
            tmp.add(nums[i]);
            dfsSubsetsWithDup(nums,ans,tmp,i + 1);
            tmp.remove(tmp.size() - 1);
        }
    }

    // 77. 组合
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> ans = new ArrayList<>();
        if (n <= 0 || k == 0)return ans;
        dfsCombine(n,k,1,ans,new ArrayDeque<>());
        return ans;
    }
    public void dfsCombine(int n, int k ,int start, List<List<Integer>> ans, Deque<Integer> deque){
        if (deque.size() == k){
            ans.add(new ArrayList<>(deque));
            return;
        }
        for (int i = start; i <= n; i++) {
            if (n - i  + deque.size() < k )continue;
            deque.add(i);
            dfsCombine(n,k,i+1,ans,deque);
            deque.removeLast();
        }
    }
    // 40. 组合总和 II
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        dfsSum2(candidates,target,0,new ArrayList<>(),ans);
        return ans;
    }
    public void dfsSum2(int[] candidate,int target,int start,List<Integer> combine, List<List<Integer>> ans){
        if (target == 0){
            ans.add(new ArrayList<>(combine));
            return;
        }
        for (int i = start; i < candidate.length; i++) {
            if (candidate[i] <= target){
                if (i > start && candidate[i] == candidate[i - 1])
                    continue;
                combine.add(candidate[i]);
                target -= candidate[i];
                dfsSum2(candidate,target,i+1,combine,ans);
                target += candidate[i];
                combine.remove(combine.size() - 1);
            }

        }
    }
    //47. 全排列 II
    public static List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        boolean[] visited = new boolean[nums.length];
        dfsPermuteUnique(nums,ans, new ArrayList<>(),visited);
        return ans;
    }
    public static void dfsPermuteUnique(int[] nums, List<List<Integer>> ans, List<Integer> combine, boolean[] visited){
        if (combine.size() == nums.length){
            ans.add(new ArrayList<>(combine));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) continue;
            if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1])continue;
            visited[i] = true;
            combine.add(nums[i]);
            dfsPermuteUnique(nums,ans,combine,visited);
            visited[i] = false;
            combine.remove(combine.size() - 1);
        }
    }
    // 46. 全排列
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        int[] visited = new int[nums.length];
        dfsPerumte(nums,new ArrayList<>(),ans,visited);
        return ans;
    }
    public void dfsPerumte(int[] nums,List<Integer> tmp,List<List<Integer>> ans,int[] visited){
        if (tmp.size() == nums.length){
            ans.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i] == 1)continue;
            visited[i] = 1;
            tmp.add(nums[i]);
            dfsPerumte(nums,tmp,ans,visited);
            visited[i] = 0;
            tmp.remove(tmp.size() - 1);
        }
    }
    //def backward():
    //
    //    if (回朔点）：# 这条路走到底的条件。也是递归出口
    //        保存该结果
    //        return
    //
    //    else:
    //        for route in all_route_set :  逐步选择当前节点下的所有可能route
    //
    //            if 剪枝条件：
    //                剪枝前的操作
    //                return   #不继续往下走了，退回上层，换个路再走
    //
    //            else：#当前路径可能是条可行路径
    //
    //                保存当前数据  #向下走之前要记住已经走过这个节点了。例如push当前节点
    //
    //                self.backward() #递归发生，继续向下走一步了。
    //
    //                回朔清理     # 该节点下的所有路径都走完了，清理堆栈，准备下一个递归。例如弹出当前节点
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        ArrayList<Integer> combine = new ArrayList<>();
        dfs(candidates,target,0,combine,ans);
        return ans;
    }
    public void dfs(int[] candidates,int target, int index, List<Integer> combine,List<List<Integer>> ans ){
        if (target < 0)return;
        if (target == 0){
            ans.add(new ArrayList(combine));
            return;
        }
        for (int start = index; start < candidates.length; start++) {
            if (target - candidates[start] < 0)break; //减枝
            combine.add(candidates[start]);
            dfs(candidates,target - candidates[start],start,combine,ans);
            combine.remove(combine.size() - 1);
        }


    }
    //
    Map<Integer,Integer> map = new HashMap<>();
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        for(int i = 0 ; i < inorder.length;i++)
            map.put(inorder[i], i); //加入map
        return buildCore(0,0,preorder.length - 1,preorder);
    }
    public TreeNode buildCore(int root,int left,int right, int[] preorder ){
        if(left > right)return null;
        TreeNode root1 = new TreeNode(preorder[root]); //构建根节点
        int tmp = map.get(preorder[root]); //获取根节点在中序遍历中对应位置
        root1.left =   buildCore(root + 1,left,tmp - 1,preorder);//开启左子树
        root1.right =  buildCore(root + 1 + tmp - left,tmp + 1,right,preorder);//右子树
        return root1;
    }

    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < tokens.length; i++) {
                if (tokens[i].equals("+"))
                    stack.push(stack.pop() + stack.pop());
                else  if (tokens[i].equals("-"))
                    stack.push(-stack.pop() + stack.pop());
                else  if (tokens[i].equals("*"))
                    stack.push(stack.pop() * stack.pop());
                else  if (tokens[i].equals("/")) {
                    int num2 = stack.pop();
                    int num1 = stack.pop();
                    stack.push(num1 / num2);
                }
                else  stack.push(Integer.parseInt(tokens[i]));
            }
       return stack.pop();
    }
    // 3. 无重复字符的最长子串
    public static int lengthOfLongestSubstring(String s) {
        if(s.length() == 0 || s == null)return 0;
        HashSet<Character> set = new HashSet<>();
        int right = -1, ans = 0;
        char[] chars = s.toCharArray();
        for (int i = 0; i < s.length(); i++) {
            if (i != 0)
                set.remove(chars[i - 1]);
            while (right + 1 < s.length() && !set.contains(chars[right + 1])){
                set.add(chars[right + 1]);
                right++;
            }
            ans = Math.max(ans, right + 1 - i);
        }
        return ans;
    }
    // 2. 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = new ListNode();
        ListNode pre = head;
        int tmp = 0;
        while (l1 != null && l2 != null){
           int sum = l1.val + l2.val + tmp;
           tmp = sum / 10;
           pre.next = new ListNode(sum % 10);
           pre = pre.next;
           l1 = l1.next;
           l2 = l2.next;
       }
        l1 = l1 == null ? l2 : l1;
        while (l1 != null){
            int sum = l1.val + tmp;
            tmp = sum / 10;
            pre.next = new ListNode(sum % 10);
            pre = pre.next;
            l1 = l1.next;
        }
        if (tmp != 0){
            pre.next = new ListNode(tmp);
        }
        return head;
    }
    // 64. 最小路径和
    public int minPathSum(int[][] grid) {
        if (grid == null || grid[0].length == 0)return -1;
        for (int i = 1; i < grid[0].length; i++) {
            grid[0][i] += grid[0][i - 1];
        }
        for (int i = 1; i < grid.length; i++) {
            grid[i][0] += grid[i - 1][0];
        }
        for (int i = 1; i < grid.length; i++) {
            for (int j= 1; j < grid[0].length; j++) {
                grid[i][j] += Math.min(grid[i][j - 1], grid[i - 1][j]);
            }
        }
        return grid[grid.length - 1][grid[0].length - 1];
    }
    // 238. 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int[] rs = new int[nums.length];
        int left = 1;int right = 1;
        for (int i = 0; i < nums.length; i++) {
            rs[i] = left;
            left *= nums[i]; // 计算 下一位(i+1）的左边的乘积
        }
        for (int i = nums.length - 1; i >= 0; i--) {
            rs[i] *= right; // i的左边乘右边
            right *= nums[i];
        }
        return rs;
    }
    // 114. 二叉树展开为链表
    public void flatten(TreeNode root) {
        LinkedList<TreeNode> stack = new LinkedList<>();
        stack.push(root);
        TreeNode tmp = null;
        TreeNode curr = root;
        while (!stack.isEmpty() && (tmp = stack.pop()).right != curr){
               tmp = stack.pop();
               if (tmp.right != null)stack.push(tmp.right);
               if (tmp.left != null)stack.push(tmp.left);
               curr.right = tmp;
               curr.left = null;
               curr = curr.right;
        }
    }
    // 287. 寻找重复数
    public int findDuplicate(int[] nums) {
        int slow = nums[0], fast = nums[0];
        while (slow != fast){
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        slow = 0;
        while (slow != fast){
            slow = nums[slow];
            fast = nums[fast];
        }
        return fast;
    }


    // n48. 旋转图像
    public void rotate(int[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                int tmp = matrix[j][i];
                matrix[j][i] = matrix[i][j];
                matrix[i][j] = tmp;
            }
        }
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length / 2; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[i][matrix[0].length - j];
                matrix[i][matrix[0].length - j] = tmp;
            }
        }
    }
    // 94. 二叉树的中序遍历
    private List<Integer> rs = new ArrayList<Integer>();
    public List<Integer> inorderTraversal(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        while (root != null || !stack.isEmpty()){
            if (root.left != null){
                stack.push(root.left);
                root = root.left;
            }else {
                TreeNode tmp = stack.pop();
                rs.add(tmp.val);
                root = tmp.right;
            }
        }
//        if (root == null) return rs;
//        if (root.left != null)inorderTraversal(root.left);
//        rs.add(root.val);
//        if (root.right != null)inorderTraversal(root.right);

        return rs;
    }
    // 338. 比特位计数
    public int[] countBits(int num) {
        int[] result = new int[num  + 1];
        for (int i = 1; i <= num; i++) {
            // i & (i - 1) 把最右边的1去掉,所以i&(i - 1) 比i要小，
            result[i] = result[i & (i - 1)] + 1;
        }
        return result;
    }
    // 461. 汉明距离
    public int hammingDistance(int x, int y) {
        int xor = x ^ y;
        int sum = 0;
        while (xor != 0){
            if ((xor & 1) ==1)
                sum++;
            xor = xor >>1;
        }
        return sum;
    }
    // 剑指 Offer 35. 复杂链表的复制
    public Node copyRandomList(Node head) {
        HashMap<Node, Node> map = new HashMap<>();
        for ( Node p = head; p != null; p = p.next){
            map.put(p, new Node(p.val));
        }
        Node curr = null;Node p = null;
        for ( p = head; p != null; p = p.next, curr = curr.next){
            curr = map.get(p);
            curr.next = map.get(p.next);
            curr.random = map.get(p.random);
        }
        return curr;
    }
    // 面试题34. 二叉树中和为某一值的路径
    public List<List<Integer>> pathSum(TreeNode root, int target) {

        List<TreeNode> path = new LinkedList<>();
        List<List<Integer>> rs = new ArrayList<>();
        pathSumCore(root, target,0, path, rs);
        return rs;
    }
    public void pathSumCore(TreeNode root, int target, int curr, List<TreeNode> path, List<List<Integer>> rs ){
        if (root == null)return;
        curr += root.val;
        path.add(root);
        if (curr == target && root.left == null && root.right == null){
            ArrayList<Integer> product = new ArrayList<>();
            for (TreeNode node : path) {
                product.add(node.val);
            }
            rs.add(product);
        }
        else {
            if (root.left != null)pathSumCore(root.left, target, curr,path, rs);
            if (root.right != null)pathSumCore(root.right, target, curr,path, rs);
        }
        path.remove(path.size() - 1);
    }
    // 剑指 Offer 10- I. 斐波那契数列
    public int fib(int n) {
        int[] mem = new int[n + 1];
        return fibCore( n , mem);
    }
    public int fibCore(int n , int[] mem){
        if (mem[n] > 0)return mem[n];
        if (n == 0)mem[0] = 0;
        if (n == 1)mem[1] = 1;
        else mem[n] =  fibCore(n - 2,mem) % 1000000007 + fibCore(n - 1, mem) % 1000000007;
        return mem[n];
    }
    // 剑指 Offer 33. 二叉搜索树的后序遍历序列
    public static boolean verifyPostorder(int[] postorder) {
        if (postorder == null || postorder.length == 0)return false;
        return verifyPostorderCore(postorder, 0,postorder.length - 1);
    }
    public static Boolean verifyPostorderCore(int[] postorder, int from, int to){
        int root = postorder[to];
        int i = from;
        for (; i < to ; i++) {
            if (postorder[i] > root)break;
        }
        int j = i;
        for (; j < to ; j++) {
            if (postorder[j] < root)return false;
        }
        Boolean left = true, right = true;
        if (i > 0)
            left = verifyPostorderCore(postorder, 0, i - 1);

        if (i < to - 1)
            right = verifyPostorderCore(postorder, i,to - 1 );
        return left && right;
    }
    // 103. 二叉树的锯齿形层序遍历
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null)return null;
        LinkedList<TreeNode>[] stacks = new LinkedList[2];
        stacks[0] = new LinkedList<>();
        stacks[1] = new LinkedList<>();
        int curr = 0;
        int next = 1;
        ArrayList<List<Integer>> rs = new ArrayList<>();
        stacks[curr].push(root);
        List<Integer> product = new ArrayList<>();
        while (!stacks[0].isEmpty() || !stacks[1].isEmpty()) {
                TreeNode tmp = stacks[curr].pop();
                product.add(tmp.val);
                if (curr == 0){
                    if (tmp.left != null)stacks[next].push(tmp.left);
                    if (tmp.right != null)stacks[next].push(tmp.right);

                }else {
                    if (tmp.right != null)stacks[next].push(tmp.right);
                    if (tmp.left != null)stacks[next].push(tmp.left);
                }
                if (stacks[curr].isEmpty()){
                    curr = 1 - curr;
                    next = 1 - next;
                    rs.add(product);
                    product = new ArrayList<>();
                }
        }
        return rs;
    }
    // 78. 子集
    public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> rs = new ArrayList<List<Integer>>();
        ArrayList<Integer> listNull = new ArrayList<>();
        rs.add(listNull);
        for (int num : nums) {
            List<Integer> tmp = null;
            int range = rs.size();
            for (int i = 0; i < range; i++) {
                tmp = new ArrayList<>(rs.get(i));
                tmp.add(num);
                rs.add(tmp);
            }
        }
        return rs;
    }
    // 剑指 Offer 46. 把数字翻译成字符串
    public int translateNum(int num) {
        if (num < 10) return 1;
        int re = num % 100;
        if (re < 10) return translateNum(num / 10);
        else if (re < 26) return translateNum(num / 100) + translateNum(num / 10);
        else return translateNum(num / 10);
    }

    // 剑指 Offer 12. 矩阵中的路径
    public boolean existPath(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0)return false;
        int rows = board.length;
        int cols = board[0].length;
        int length = 0;
        boolean[] visited = new boolean[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {// 相等且没有访问过
                if (existCore(i, j, board, length, word, visited))
                    return true;
            }
        }
        return false;
    }
    public boolean existCore(int row, int col, char[][] board, int length, String word, boolean[] visited){
        if (length == word.length())return true;
        int rows = board.length;
        int cols = board[0].length;
        Boolean hashNext = false;
        if(row > -1 && col > -1 && row < rows && col < cols && !visited[row * cols + col] && board[row][col] ==word.charAt(length)) {
            length++;
            visited[row * cols + col] = true;
             hashNext = existCore(row - 1, col, board, length, word, visited)
                    || existCore(row + 1, col, board, length, word, visited)
                    || existCore(row, col - 1, board, length, word, visited)
                    || existCore(row, col + 1, board, length, word, visited);
            if (!hashNext){
                length--;
                visited[row * cols + col] = false;
            }
        }
        return hashNext;
    }
    // 剑指 Offer 14- I. 剪绳子
    public int cuttingRope(int n) {
        if (n == 1)return 0;
        if (n == 2)return 1;
        if (n == 3)return 2;
        int[] ints = new int[n + 1]; // 长度为n，只能取到ints[n - 1]   长度为n + 1 能取到ints[n]
        ints[0] = 0;
        ints[1] = 0;
        ints[2] = 1;
        ints[3] = 2;
        for (int i = 4; i <= n; i++) { // i 指代绳子还有的长度
            int max = 0;
            for (int j = 1; j <= i / 2; j++) { // j 指代切j长度，剩下i-j长度
                int tmp = ints[j] * ints[i - j];
                if (max < tmp){
                    max = tmp;
                    ints[i] = max;  // 绳子长度为i时的最大值
                }
            }
        }
        return ints[n];
    }
    // 剑指 Offer 59 - I. 滑动窗口的最大值
    public static int[] maxSlidingWindow(int[] nums, int k) {
        ArrayList<Integer> rs = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            int max = Integer.MIN_VALUE, j = 0 ;
            for (j = i + k - 1; j >= i && j < nums.length; j--) {
                max = max < nums[j] ? nums[j] : max;
            }
            if (j < nums.length)rs.add(max);
        }
        int[] a = new int[rs.size()];
        for (int i = 0; i < rs.size(); i++) {
            a[i] = rs.get(i);
        }
        return a;
    }
    // 剑指 Offer 53 - II. 0～n-1中缺失的数字
    public int missingNumber(int[] nums) {
        int i = 0;
        while (i < nums.length){
            if (nums[i] == i)i++;
            else break;
        }
        return i;
    }
    // 654. 最大二叉树
    public static TreeNode constructMaximumBinaryTree(int[] nums) {
        return searchTree(0, nums.length - 1, nums);
    }
    public static TreeNode searchTree(int left, int right, int[] nums){
        int max = Integer.MIN_VALUE;
        int index = 0;
        if (left > right)return null;
        for (int i = left; i <= right; i++) {
            if (max < nums[i]){
                max = nums[i];
                index = i;
            }
        }
        TreeNode root = new TreeNode(nums[index]);
        root.left = searchTree(left, index - 1,nums);
        root.right = searchTree(index + 1, right, nums);
        return root;
    }
    // 剑指 Offer 58 - I. 翻转单词顺序
    public static String reverseWords(String s) {
        StringBuilder builder = new StringBuilder();
        int r = s.length() - 1;
        int l = r;
        while (l >= 0){
            while (l >= 0 && s.charAt(l) != ' ')l--;
            builder.append(s.substring(l + 1, r + 1) + " ");
            while (l >= 0 && s.charAt(l) == ' ')l--;
            r = l;
        }
        return builder.toString().trim();
//        String trim = s.trim();
//        String[] strings = trim.split(" ");
//        String rs = "";
//        for (int i = strings.length - 1; i > 0 ; i--) {
//            if (!strings[i].isEmpty())
//                rs += strings[i].trim() + " ";
//        }
//        rs += strings[0];
//        return rs;
    }
    // 剑指 Offer 04. 二维数组中的查找
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0)return false;
        int j = matrix[0].length - 1;
        for (int i = 0; i < matrix.length; i++) {
            while (j >= 0 && target <= matrix[i][j]) {
                if (target == matrix[i][j])return true;
                j--;
            }
        }
        return false;
    }
    // 剑指 Offer 31. 栈的压入、弹出序列
    public static boolean validateStackSequences(int[] pushed, int[] popped) {
        if (pushed.length == 0 || popped.length == 0)return true;
        ArrayDeque<Integer> deque = new ArrayDeque<>();
        deque.push(pushed[0]);
        int i = 1, j = 0;
        while (!deque.isEmpty()){
            while (!deque.isEmpty() && deque.peek() == popped[j]){
               // if (deque.isEmpty())return true;
                deque.poll();
                j++;

            }
            if(deque.isEmpty() && i >= pushed.length)return true;
            if (i < pushed.length){
                deque.push(pushed[i]);
                i++;
            }
            else return false;
        }
        return true;
    }
    // 剑指 Offer 10- II. 青蛙跳台阶问题
    public int numWays(int n) {
        int[] nums = new int[n + 1];
        return jump(n, nums);
    }
    public int jump(int n, int[] nums){
        if (nums[n] > 0)return nums[n]; // 保证已经计算过的不被重复计算
        if (n == 1)nums[n] = 1;
        else if (n == 2)nums[n] = 2;
        else nums[n] = (jump(n - 1,nums) + jump(n - 2, nums)) % 1000000007;
        return nums[n];
    }
    // 剑指 Offer 09. 用两个栈实现队列
    class CQueue {
        private Stack s1 ;
        private Stack s2;
        public CQueue() {
            this.s1 = new Stack<Integer>();
            this.s2 = new Stack<Integer>();
        }
        public void appendTail(int value) {
            if (!s2.isEmpty()){
                while (!s2.isEmpty()){
                    s1.push(s2.pop());
                }
            }
            s1.push(value);
        }

        public int deleteHead() {

            while (!s1.isEmpty()){
                s2.push(s1.pop());
            }
            if (s1.isEmpty())return -1;
            else return (int)s2.pop();
        }
    }
    // 1379. 找出克隆二叉树中的相同节点
    public final TreeNode getTargetCopy(final TreeNode original, final TreeNode cloned, final TreeNode target) {
        TreeNode tmp_o = original;
        TreeNode tmp_c = cloned;
        if (tmp_o == null || tmp_c == null) return null;
        if (tmp_o.val == target.val)return tmp_c;
        TreeNode left = getTargetCopy(tmp_o.left, tmp_c.left, target);
        if (left != null)return left;
        return getTargetCopy(tmp_o.right, tmp_c.right, target);

    }
    // 1302. 层数最深叶子节点的和
    public int dfs(TreeNode root , int dep, int maxdep, int sum){
        while (root != null){
            dep++;
            if (maxdep < dep){
                maxdep = dep;
                sum = root.val;
            }
            sum += root.val;
            if (root.left != null)dfs(root.left, dep, maxdep, sum);
            if (root.right != null)dfs(root.right, dep, maxdep, sum);
        }
        return sum;
    }
    public int deepestLeavesSum(TreeNode root) {
        // 深度遍历
        return dfs(root, 0, -1, 0);

        //层序遍历
//        if (root == null)return -1;
//        Queue<TreeNode> queue = new LinkedList<>();
//        queue.add(root);
//        int sum = 0, size = 0;
//        while (!queue.isEmpty()){
//            sum = 0;
//            size = queue.size();
//            for (int i = 0; i < size; i++) {
//                TreeNode tmp = queue.poll();
//                sum += tmp.val;
//                if (tmp.left != null)queue.add(tmp.left);
//                if (tmp.right != null)queue.add(tmp.right);
//            }
//        }
//        return sum;
    }
    // 面试题 04.02. 最小高度树
    public TreeNode sortedArrayToBST1(int[] nums) {
            return middle(0, nums.length - 1, nums);
    }
    public TreeNode middle(int start, int end, int[] nums){
        if (start < end){
            int mid = (end - start) / 2;
            TreeNode node = new TreeNode(nums[mid]);
            node.left = middle(start, mid - 1, nums);
            node.right = middle(mid + 1, end, nums);
            return node;
        }
        return null;
    }
    // 219. 存在重复元素 II
    public static boolean containsNearbyDuplicate(int[] nums, int k) {
        for (int i = 0; i < nums.length; i++) {
            
        }
        return false;
    }
    // 217. 存在重复元素
    public boolean containsDuplicate(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num))return true;
            set.add(num);
        }
        return false;
    }
    // 645. 错误的集合
    public static int[] findErrorNums(int[] nums) {
        HashSet<Integer> set = new HashSet<>();
        int congfu = 0, que = 0;
        for (Integer num : nums) {
            if (set.contains(num)){
                congfu = num;
            }
            set.add(num);
        }
        for (int i = 1; i <= nums.length; i++) {
            if (!set.contains(i))
            {
                que = i;
                break;
            }
        }
        return new int[]{congfu, que};
    }
    // 剑指 Offer 53 - I. 在排序数组中查找数字 I   5 7 7 8 8 10   8
    public static int search(int[] nums, int target) {
        if(nums.length == 0 || nums == null)return 0;
        int sum = 0, mid = nums.length / 2, low = 0, high = nums.length - 1;
        while (low < high){
            if (nums[mid] == target)
                break;
            else if (nums[mid] > target)high = mid - 1;
            else low = mid + 1;
            mid = (low + high) / 2;
        }
        if (nums[mid] == target){
            sum++;
            int k = mid - 1;
            while (k >= 0 && (nums[k] == target) ){
                sum++;
                k--;
            }
            k = mid + 1;
            while (k < nums.length && (nums[k] == target)){
                sum++;
                k++;
            }
            return sum;
        }
        return sum;

    }
    //a b ap ba app ban appl bana apply banan banana
    public static String longestWord(String[] words) {
        HashSet<String> set = new HashSet<>();
        for (String word : words) set.add(word);
        Arrays.sort(words, (x, y) -> x.length() == y.length() ? x.compareTo(y) : y.length() - x.length()) ;
        for (String word : words) {
            boolean flag = true;
            for (int i = 1; i < word.length(); i++) {
                if (!set.contains(word.substring(0,i))){
                    flag = false;
                    break;
                }
            }
            if (flag)return word;
        }
        return "";
    }
//    // 剑指 Offer 38. 字符串的排列
//    public String[] permutation(String s) {
//        char[] chars = s.toCharArray();
//        List<String> rs = new ArrayList<>();
//        dfs(chars[0]);
//
//        return rs.toArray(new String[rs.size()]);
//    }
//    public void dfs(int root, char[] chars){
//        if (root == chars.length - 1)
//    }
//    public void  swap(int a[], int b[], int index_a, int index_b){
//        int tmp = b[index_b];
//        b[index_b] = a[index_a];
//        a[index_a] = tmp;
//    }
    // 剑指 Offer 45. 把数组排成最小的数
    public String minNumber(int[] nums) {
        String[] arr =  new String[nums.length];
        for (int i = 0; i < arr.length; i++){
            arr[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(arr, (x, y) -> (x + y ).compareTo(y + x));
        StringBuilder rs = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            rs.append(arr[i]);
        }
        return rs.toString();
    }
    // 剑指 Offer 64. 求1+2+…+n
    public int sumNums(int n) {
        boolean rs = n > 0 && (n +=sumNums(n - 1) ) > 0;
        return n;
    }
    // 剑指 Offer 42. 连续子数组的最大和
    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (dp[i - 1] > 0){
                dp[i] = dp[i - 1] + nums[i];
            }else
                dp[i] = nums[i];
        }
        int max = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            if (max < dp[i])max = dp[i];
        }
        return max;
    }
    // 387. 字符串中的第一个唯一字符
    public int firstUniqChar1(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            map.put(chars[i], map.getOrDefault(chars[i], 0) + 1);
        }
        for (int i = 0; i < chars.length; i++) {
            if (map.get(chars[i]) == 1)return i;
        }
        return -1;
    }
    // 594. 最长和谐子序列
    public int findLHS(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        int length = 0, key_1 = 0, key_2 = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            for (Map.Entry<Integer, Integer> innerentry : map.entrySet()) {
                if (Math.abs(entry.getKey() - innerentry.getKey()) == 1)
                        length =  (entry.getValue() + innerentry.getValue()) > length ? (entry.getValue() + innerentry.getValue()) : length;
            }
        }
        return length;
    }
    // 205. 同构字符串
    public boolean isIsomorphic(String s, String t) {
        HashMap<Character, Character> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))){
                if (map.get(s.charAt(i)) != t.charAt(i))return false;
            }else {
                if (map.containsValue(s.charAt(i)))return false;
                map.put(s.charAt(i), t.charAt(i));
            }
        }
        return true;
    }
    //290. 单词规律
    public static boolean wordPattern(String pattern, String s) {
        if (pattern == "" || s == "")return false;
        char[] split = pattern.toCharArray();
        String[] split_s = s.split(" ");
        if (pattern.length() != split_s.length)return false;
        HashMap<Character, String> map = new HashMap<>();
        for (int i = 0; i < split.length; i++) {
            if (!map.containsKey(split[i])){
                if (map.containsValue(split_s[i]))return false;
                map.put(split[i], split_s[i]);
            }else if (!map.get(split[i]).equals(split_s[i]))return false;
        }
        return true;
    }
    // 面试题 01.04. 回文排列
    public static boolean canPermutePalindrome(String s) {
        char[] tmp = s.toCharArray();
        HashMap<Character, Integer> map = new HashMap<>();
        for (char c : tmp)
            map.put(c, 1 + map.getOrDefault(c, 0));
        int ji = 0;
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            if (entry.getValue() % 2 == 1)ji++;
        }
        return (ji == 1 || ji == 0 )? true : false;
    }
    //1189. “气球” 的最大数量
    public static int maxNumberOfBalloons(String text) {
        int[] chars = new int[32];
        char[] array = text.toCharArray();
        for (int i = 0; i < array.length; i++) {
            chars[array[i] - 'a'] ++;
        }
        int[] ints = new int[5];
         ints[0] = chars['b' - 'a'];
         ints[1] = chars[0];
         ints[2] = chars['l' - 'a'] / 2;
         ints[3] = chars['o' - 'a'] / 2;
         ints[4] = chars['n' - 'a'];
         int min = ints[0];
        for (int i = 0; i < 5; i++) {
            if (ints[i] == 0)return 0;
            if (min > ints[i])min = ints[i];
        }
        return min;

    }
    //884. 两句话中的不常见单词
    public String[] uncommonFromSentences(String A, String B) {
        String[] a = A.split(" ");
        String[] b = B.split(" ");
        HashMap<String, Integer> map = new HashMap<>();
        for (int i = 0; i < a.length; i++) {
                map.put(a[i], map.getOrDefault(a[i], 0) + 1);
        }
        for (int i = 0; i < b.length; i++) {
            map.put(b[i], map.getOrDefault(b[i], 0) + 1);
        }
        ArrayList<String> list = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            if (entry.getValue() == 1)list.add(entry.getKey());
        }
        return list.toArray(new String[list.size()]);
    }
    //961. 重复 N 次的元素
    public int repeatedNTimes(int[] A) {
        int n = A.length / 2;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < A.length; i++) {
            map.put(A[i], map.getOrDefault(A[i], 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry: map.entrySet()) {
            if (entry.getValue() == n)return entry.getKey();
        }
        return -1;
    }
    // 389. 找不同
    public static char findTheDifference(String s, String t) {
        int rs = 0;
        for (char c : s.toCharArray()) {
            rs ^= c;
        }
        for (char c : t.toCharArray()) {
            rs ^= c;
        }
        return (char) rs;
    }
    // 1640. 能否连接形成数组
    public boolean canFormArray(int[] arr, int[][] pieces) {
        HashMap<Integer, int[]> map = new HashMap<>();
        for (int[] piece : pieces) {
            map.put(piece[0], piece);
        }
//        [91,4,64,78]
//        [[78],[4,64],[91]]
        for (int i = 0; i < arr.length; i++) {
            if (map.containsKey(arr[i])){
                int[] ints = map.get(arr[i]);
                for (int j = 1, k = i + 1; j <ints.length;k++, j++) {
                    if (arr[k] != ints[j])return false;
                    i = k;
                }
            }
            else return false;
        }
        return true;
    }
    // 剑指 Offer 57 - II. 和为s的连续正数序列
    public int[][] findContinuousSequence(int target) {
        List<int[]> arr = new ArrayList<>();
       int limit = (target - 1) / 2;
       int sum = 0 ;
        for (int i = 0; i < limit; i++) {
            for (int j = i;  ; j++) {
                sum += j;
                if (sum == target){
                    int[] rs = new int[j - i + 1];
                    for (int k = i; k <= j; k++) {
                        rs[k - i] = k;
                    }
                    arr.add(rs);
                    break;
                }else if (sum > target ){
                    sum = 0;
                    break;
                }
            }
        }
        return arr.toArray(new int[arr.size()][]);
    }
    // 剑指 Offer 66. 构建乘积数组
    public int[] constructArr(int[] a) {
        int[] left = new int[a.length];
        int[] right = new int[a.length];
        int[] rs = new int[a.length];
        left[0] = right[right.length - 1] = 1;
        for (int i = 1; i < left.length; i++) {
            left[i] = left[i - 1] * a[i - 1];
        }
        for (int i = right.length - 2 ; i >= 0; i--) {
            right[i] = right[i + 1] * a[i + 1];
        }
        for (int i = 0; i < rs.length; i++) {
            rs[i] = left[i] * right[i];
        }
        return rs;
    }
    // 剑指 Offer 56 - II. 数组中数字出现的次数 II
    public static int singleNumber1(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i],0)+1);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() == 1)return entry.getKey();
        }
        return -1;
    }
    //剑指 Offer 47. 礼物的最大价值
    public int maxValue(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 1; i < n; i++) {
            grid[0][i] += grid[0][i - 1];
        }
        for (int i = 1; i < m; i++) {
            grid[i][0] += grid[i - 1][0];
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                grid[i][j] += Math.max(grid[i][j - 1], grid[i - 1][j]);
            }
        }

        return grid[m-1][n-1];

    }
    // 575. 分糖果
    public static int distributeCandies(int[] candyType) {
            int mid = candyType.length / 2;
            Set<Integer> set = new HashSet<>();
            for (int i = 0; i < candyType.length; i++) {
                set.add(candyType[i]);
            }
            if (set.size() > mid)return mid;
            return set.size();
    }
    // 剑指 Offer 63. 股票的最大利润
    public int maxProfit(int[] prices) {
        if (prices.length == 0)return 0 ;
        int dp = Integer.MIN_VALUE;
        int minPrice = Integer.MAX_VALUE;
        for (int i = 0; i < prices.length; i++) {
            minPrice = Math.min(minPrice, prices[i]);
            dp = Math.max(dp, prices[i] - minPrice);
        }
        return dp;
    }
    //剑指 Offer 28. 对称的二叉树
    public boolean isSymmetric(TreeNode root) {
        return mirror(root, root);
    }
    public boolean mirror(TreeNode left, TreeNode right){
        if (left == null && right == null)return true;
        if (left == null || right == null)return false;
        if (left.val == right.val)
            return mirror(left.left, right.right) && mirror(left.right, right.left);
        return false;
    }
    //剑指 Offer 55 - II. 平衡二叉树
    public boolean isBalanced(TreeNode root) {
        return height(root) != -1 ;
    }
    public int height(TreeNode root){
        if (root == null) return 0;
        int left = height(root.left);
        if (left == -1)return -1;
        int right = height(root.right);
        if (right == -1)return -1;
        return Math.abs(left - right) <= 1 ? Math.max(left, right) + 1: -1 ;
    }

    //剑指 Offer 32 - III. 从上到下打印二叉树 III
    public static List<List<Integer>> levelOrder12(TreeNode root) {
        List<List<Integer>> rs = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if(root != null) queue.add(root);
        while (!queue.isEmpty()){
            LinkedList<Integer> temp = new LinkedList<>();
            for(int i = queue.size(); i > 0; i--)  {
                TreeNode node = queue.poll();
                if (rs.size() % 2 == 0)temp.addLast(node.val);
                else temp.addFirst(node.val);
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            rs.add(temp);
        }
        return rs;
    }
    //
    public int[] getLeastNumbers(int[] arr, int k) {
        int[] vec = new int[k];
        if (k == 0) { // 排除 0 的情况
            return vec;
        }
        PriorityQueue<Integer> queue = new PriorityQueue<Integer>(new Comparator<Integer>() {
            public int compare(Integer num1, Integer num2) {
                return num2 - num1;
            }
        });
        for (int i = 0; i < k; ++i) {
            queue.offer(arr[i]);
        }
        for (int i = k; i < arr.length; ++i) {
            if (queue.peek() > arr[i]) {
                queue.poll();
                queue.offer(arr[i]);
            }
        }
        for (int i = 0; i < k; ++i) {
            vec[i] = queue.poll();
        }
        return vec;
    }
    // 1160. 拼写单词
    public static int countCharacters(String[] words, String chars) {
        HashMap<Character, Integer> map_word = new HashMap<>();
        HashMap<Character, Integer> map_char = new HashMap<>();
        int length = 0;
        for (char elm : chars.toCharArray()) {
            map_char.put(Character.valueOf(elm), map_char.getOrDefault(Character.valueOf(elm), 0 ) + 1);
        }

        for (String word : words) {
            int temp = 0;
            for (char c : word.toCharArray())
                map_word.put(Character.valueOf(c), map_word.getOrDefault(Character.valueOf(c), 0 ) + 1);
            for (Map.Entry<Character, Integer> entry : map_word.entrySet()) {
                if (!map_char.containsKey(entry.getKey()) || map_char.get(entry.getKey()) < entry.getValue()){
                    temp = 0;
                    break;
                }
                temp += entry.getValue();
            }
            length += temp;
            map_word.clear();
        }
        return length;
    }

    //剑指 Offer 18. 删除链表的节点
    public ListNode deleteNode(ListNode head, int val) {
        ListNode fakehead = new ListNode(-1);
        fakehead.next = head;
        ListNode pre = fakehead;
        ListNode temp = pre.next;
        while (temp != null){
            if (temp.val == val){
                pre.next = temp.next;
                temp = pre.next;
            }
            pre = temp;
            temp = pre == null ? null : pre.next;
        }
        return fakehead.next;
    }
    //剑指 Offer 50. 第一个只出现一次的字符
    public char firstUniqChar(String s) {
        LinkedHashMap<Character,Boolean> map = new LinkedHashMap<>();
        for (int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i),!map.containsKey(s.charAt(i)));
        }
        for (Map.Entry<Character, Boolean> entry : map.entrySet()) {
            if (entry.getValue())return (char) entry.getKey();
        }
       return ' ';
    }
    //剑指 Offer 52. 两个链表的第一个公共节点
    public ListNode getIntersectionNode12(ListNode headA, ListNode headB) {
        int length = 0;
        ListNode temp_a = headA;
        ListNode temp_b = headB;
        while (temp_a != null){
            length++;
            temp_a = temp_a.next;
        }
        while (temp_b != null){
            length--;
            temp_b = temp_b.next;
        }
        //把长的链表给A
        temp_a = length < 0 ? headB : headA;
        temp_b = length < 0 ? headA : headB;
        //让长的先走
        length = Math.abs(length);
        while (length != 0) {
            temp_a = temp_a.next;
            length--;
        }
        while (temp_a != null){
            if (temp_a == temp_b)return temp_b;
            temp_a = temp_a.next;
            temp_b = temp_b.next;
        }
        return null;
    }
    //剑指 Offer 15. 二进制中1的个数
    public int hammingWeight(int n) {
        int sum = 0;
        while (n != 0){
            if ((n & 1) == 1){
                sum++;
                n >>>= 1;
            }
        }
        return sum;
    }
    //剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
    public int[] exchange(int[] nums) {
        int pre = 0, post = nums.length - 1;
        while (pre < post){
            while (pre < post && (nums[pre] & 1) == 0)pre++;
            while (pre < post && (nums[post] & 1) == 1)post--;
            int tmp = nums[pre];
            nums[pre] = nums[post];
            nums[post] = tmp;
        }
        return nums;
    }
    //剑指 Offer 32 - I. 从上到下打印二叉树
    public int[] levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> rs = new ArrayList<>();
        while (!queue.isEmpty()){
            TreeNode temp = queue.poll();
            rs.add(temp.val);
            if (temp.left != null)queue.add(temp.left);
            if (temp.right != null)queue.add(temp.right);
        }
        int[] ints = new int[rs.size()];
        for (int i = 0; i < rs.size(); i++) {
            ints[i] = rs.get(i);
        }
        return ints;
    }
    //剑指 Offer 57. 和为s的两个数字
    public static int[] twoSum(int[] nums, int target) {
        for (int i = 0, j = nums.length - 1; i < nums.length;) {
            if (nums[i] + nums[j] > target)
                j--;
            else if (nums[i] + nums[j] < target)
                i++;
            else if(nums[i] == nums[j])return new int[]{nums[i],nums[j]};
        }
        return null;
    }
    //剑指 Offer 03. 数组中重复的数字
    public int findRepeatNumber(int[] nums) {
        Arrays.sort(nums);
        int x  = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (x == nums[i])return x;
            else x = nums[i];
        }
        return -1;
    }
    //剑指 Offer 39. 数组中出现次数超过一半的数字
    public int majorityElement(int[] nums) {
       Arrays.sort(nums);
       return nums[nums.length - 1];
    }
    //剑指 Offer 27. 二叉树的镜像
    public TreeNode mirrorTree(TreeNode root) {
        if(root == null) return null;
        TreeNode temp = root.left;
        root.left = mirrorTree(root.right);
        root.right = mirrorTree(temp);
        return root;
    }
    //剑指 Offer 56 - I. 数组中数字出现的次数
    public int[] singleNumbers(int[] nums) {
        int eor = 0;
        for (int num : nums) {
            eor ^= num;
        }
        int rightOne = eor & (~eor + 1);
        int rs_1 = 0;
        for (int num : nums) {
            if((num & rightOne) == 0)
                rs_1 ^= num;
        }
        int rs_2 = rs_1 ^ eor;
        return new int[]{rs_1,rs_2};
    }
    class CQueue2 {
        private Stack s1;
        private Stack s2;
        public CQueue2() {
            this.s1 = new Stack<Integer>();
            this.s2 = new Stack<Integer>();
        }
        public void appendTail(int value) {
            while (!s1.isEmpty()){
                s2.push(s1.pop());
            }
            s1.push(value);
            while (!s2.isEmpty()){
                s1.push(s2.pop());
            }
        }

        public int deleteHead() {
            while (!s1.isEmpty()){
                s2.push(s1.pop());
            }
            int rs = -1;
            if (!s2.isEmpty())
                rs = (int) s2.pop();
            while (!s2.isEmpty())
                s1.push(s2.pop());
            return rs;
        }
    }
    //剑指 Offer 25. 合并两个排序的链表
    public ListNode mergeTwoLists1(ListNode l1, ListNode l2) {
//        ListNode temp_1 = l1,temp_2 = l2;
        ListNode head = new ListNode(-1);
        ListNode temp = head;
        while (l1 != null && l2 != null){
            if (l1.val < l2.val){
                temp.next = l1;
                temp = l1;
                l1 = l1.next;
            }
            else {
                temp.next = l2;
                temp = l2;
                l2 = l2.next;
            }
        }
        l1 = l1 == null ? l2 : l1;
        temp.next = l1;
        return head.next;
     }
    //剑指 Offer 24. 反转链表
    public ListNode reverseList1(ListNode head) {
        ListNode node = null;
        ListNode next = null;
        ListNode temp = head;
        while (temp != null){
            next = temp.next;
            temp.next = node;
            node = temp;
            temp = next;
        }
        return node;
    }
    //剑指 Offer 05. 替换空格
    public String replaceSpace(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char temp = s.charAt(i);
            if (temp == ' ')
                sb.append("%20");
            else sb.append(temp);
        }
        return sb.toString();
    }
    //剑指 Offer 17. 打印从1到最大的n位数
    public int[] printNumbers(int n) {
        int[] rs = new int[(int) Math.pow(10, n) - 1];
        for (int i = 0; i < rs.length - 1; i++) {
            rs[i] = i + 1;
        }
        return rs;
    }
    //剑指 Offer 22. 链表中倒数第k个节点
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode temp = head;
        int length = 0;
        while (temp != null){
            length++;
            temp = temp.next;
        }
        if (length < k)return null;
        temp = head;
        for (int i = 0; i < length - k ; i++) {
            temp = temp.next;
        }
        return temp;
    }
    //917. 仅仅反转字母
    public static String reverseOnlyLetters(String S) {
        char[] array = S.toCharArray();
        int i  = 0,j = S.length()-1;
        while (i < j){
            if(((array[i] >= 'A' && array[i] <= 'Z') || (array[i] >= 'a' && array[i] <= 'z'))
                    && ((array[j] >= 'A' && array[j] <= 'Z') || (array[j] >= 'a' && array[j] <= 'z'))){
                char temp = array[i];
                array[i] = array[j];
                array[j] = temp;
                i++;
                j--;
            }
            else if (((array[i] >= 'A' && array[i] <= 'Z') || (array[i] >= 'a' && array[i] <= 'z'))
                    && (array[i] < 'A' || (array[i] > 'Z' && array[i] < 'a') || array[i] > 'z')
                )
                j--;
            else if ((array[i] < 'A' || (array[i] > 'Z' && array[i] < 'a') || array[i] > 'z')
                    && ((array[i] >= 'A' && array[i] <= 'Z') || (array[i] >= 'a' && array[i] <= 'z'))
            )
                i++;
            else {
                i++;
                j--;
            }
        }
        return new String(array);
    }
    //1309. 解码字母到整数映射
    public String freqAlphabets(String s) {
        StringBuffer rs = new StringBuffer();
        int i = s.length()-1;
        while (i >= 0){
            if (s.charAt(i) == '#'){
                String str = String.valueOf(s.charAt(i-1) + s.charAt(i-2));
                int temp = Integer.valueOf(str) - 1;
                rs.append((char)(temp + 'a'));
                i -= 3;
            }else {
                int temp = Integer.valueOf(String.valueOf(s.charAt(i))) - 1;
                rs.append((char)(temp + 'a'));
                i--;
            }
        }
        return rs.reverse().toString();
    }

    //415. 字符串相加
    public static String addStrings(String num1, String num2) {
        int i = num1.length()-1, j = num2.length()-1;
        StringBuffer rs = new StringBuffer();
        int yu = 0;
        while (i >= 0 || j >= 0){
            int a1 = i >= 0 ? num1.charAt(i) - '0' : 0;
            int a2 = j >= 0 ? num2.charAt(j) - '0' : 0;
            int temp = a1 + a2 + yu;
            rs.append(temp % 10);
            yu = temp / 10;
            j--;
            i--;
        }
        if (yu != 0)rs.append(yu);
        return rs.reverse().toString();


    }
    //1221. 分割平衡字符串
    public int balancedStringSplit(String s) {
        char[] chars = s.toCharArray();
        int right = 0, left = 0, sum = 0;
        for (int i = 0; i <chars.length; i++) {
            char temp = chars[i];
            if (temp == 'R')
                right++;
            if (temp == 'L')
                left++;
            if (right == left) {
                sum++;
                right = 0;
                left = 0 ;
            }
        }
        return sum;
    }
    //1108. IP 地址无效化
    public String defangIPaddr(String address) {
        return address.replace(".","[.]");
    }
    //136. 只出现一次的数字
    public static int singleNumber(int[] nums) {
        Arrays.sort(nums);
        int same = 1;
        for (int i = 0,j = i + 1; i < nums.length - 1;  i++,j++) {
            if (nums[i] == nums[j]) {
                same++;
            }
            else {
                if (same == 1)return nums[i];
                same = 1;
            }
        }
        return nums[nums.length - 1];
    }
    // 1207. 独一无二的出现次数
    public static boolean uniqueOccurrences(int[] arr) {
        HashMap<Integer,Integer> map = new HashMap<>();
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0; i < arr.length; i++) {
            map.put(arr[i],map.getOrDefault(arr[i],0) + 1);
        }
        for (Integer val : map.values()) {
            if(!set.contains(val))
                set.add(val);
            else return false;
        }
        return true;
    }
    //1002. 查找常用字符
    public static List<String> commonChars(String[] A) {
        HashMap<String,Integer> map = new HashMap<>();
        HashMap<String,Integer> map_temp = new HashMap<>();
        if (A.length != 0){
            String str1 = A[0];
            String key = null;
            for (int i = 0; i < str1.length(); i++) {
                key = String.valueOf(str1.charAt(i));
                map.put(key, map.getOrDefault(key,0) + 1);
            }
            for (int i = 1; i < A.length; i++) {
                str1 = A[i];
                for (int j = 0; j < str1.length(); j++) {
                    key = String.valueOf(str1.charAt(j));
                    map_temp.put(key, map_temp.getOrDefault(key,0) + 1);
                }
                Iterator<String> iterator = map.keySet().iterator();
                while (iterator.hasNext()){
                    String s = iterator.next();
                    if (map_temp.containsKey(s) )
                        map.put(s, (map_temp.get(s) <= map.get(s)) ? map_temp.get(s) : map.get(s));
                    else
                        iterator.remove();
                }
                map_temp.clear();
            }
            String[] rs = new String[map.keySet().size()];
            List<String> rs_list = new ArrayList<>();
            for (String s : map.keySet()) {
                for (int i = 0; i < map.get(s); i++) {
                    rs_list.add(s);
                }
            }
           return rs_list;
        }
        return null;
    }
    //1365. 有多少小于当前数字的数字
    public int[] smallerNumbersThanCurrent(int[] nums) {
        int[] rs = new int[nums.length];
        int[] temp = new int[101];
        for (int i = 0; i < nums.length; i++) {
            temp[nums[i]] ++;
        }
        for (int i = 1; i < 101; i++) {
            temp[i] += temp[i - 1];
        }
        for (int i = 0; i < rs.length; i++) {

            rs[i] = nums[i] == 0 ? 0 : temp[nums[i] - 1];
        }
        return rs;
    }
    //771. 宝石与石头
    public int numJewelsInStones(String J, String S) {
        char[] j = J.toCharArray();
        char[] s = S.toCharArray();
        int sum = 0;
        for (int i = 0; i < j.length; i++) {
            for (int k = 0; k < s.length; k++) {
                if (j[i] == s[k])
                    sum++;
            }
        }
        return sum;
    }
    //1512. 好数对的数目
    public int numIdenticalPairs(int[] nums) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            for (int j = nums.length-1; j > i ; j--) {
                if (nums[i] == nums[j])
                    sum++;
            }
        }
        return sum;
    }
    //26. 删除排序数组中的重复项
    public int removeDuplicates(int[] nums) {
        int k = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[k] != nums[i]){
                nums[++k] = nums[i];
            }
        }
        return k+1;
    }
    //350. 两个数组的交集 II
    public static int[] intersect(int[] nums1, int[] nums2) {
        HashMap<Integer,Integer> map_1 = new HashMap<>();
        HashMap<Integer,Integer> map_2 = new HashMap<>();
        List<Integer> rs = new ArrayList<>();
        for (int i : nums1) {
            if (map_1.containsKey(i))
                map_1.put(i,map_1.get(i) + 1);
            else map_1.put(i,1);
        }
        for (int i : nums2) {
            if (map_2.containsKey(i))
                map_2.put(i,map_2.get(i) + 1);
            else map_2.put(i,1);
        }
        //把小的map给map1
        if (map_1.size() > map_2.size()){
            HashMap<Integer,Integer> temp = null;
            temp = map_2;
            map_2 = map_1;
            map_1 = temp;
        }
        for (Integer key : map_1.keySet()) {
            if (map_2.containsKey(key)){
                Integer temp = 0;
                if (map_2.get(key) >= map_1.get(key))
                    temp = map_1.get(key);
                else temp = map_2.get(key);
                for (int i = 0; i < temp; i++) {
                    rs.add(key);
                }
            }
        }
        int[] ints = new int[rs.size()];
        for (int i = 0; i < rs.size(); i++) {
            ints[i] = rs.get(i);
        }
        return ints;
    }
    //349. 两个数组的交集
    public static int[] intersection(int[] nums1, int[] nums2) {
        if(nums1.length == 0 || nums2.length == 0)return null;
        HashSet<Integer> rs_1 = new HashSet<>();
        HashSet<Integer> rs_2 = new HashSet<>();
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums1.length; i++) {
            rs_1.add(nums1[i]);
        }
        for (int i = 0; i < nums2.length; i++) {
            rs_2.add(nums2[i]);
        }
//        rs_1 = rs_1.size() >= rs_2.size() ? rs_1 : rs_2;
//        rs_2 = rs_1 == rs_1 ? rs_2 : rs_1;
        if (rs_1.size() < rs_2.size()){
            HashSet temp = rs_1;
            rs_1 = rs_2;
            rs_2 = temp;
        }
        for (Integer val : rs_1) {
            if (rs_2.contains(val))list.add(val);
        }
        Integer[] a = new Integer[list.size()];
        int[] aa = new int[list.size()];
        a = list.toArray(a);
        for (int i = 0; i < a.length; i++) {
            aa[i] = a[i];
        }
        return aa;
    }
    //977. 有序数组的平方
    public int[] sortedSquares(int[] A) {
        for (int i = 0; i < A.length; i++) {
            A[i] = A[i] * A[i];
        }
        Arrays.sort(A);
        return A;
    }
    public void quickSort(int[] A){

    }

    //344. 反转字符串
    public void reverseString(char[] s) {
        int j = s.length;
        int i = 0;
        while (i < j) {
            char temp = s[i];
            s[i] = s[j];
            s[j] = temp;
            i++;
            j--;
        }
    }
    //88. 合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        for (int i = 0,j = m; j < n+m && i < n; i++,j++) {
            nums1[j] = nums2[i];
        }
        Arrays.sort(nums1,0,m+n);
//        ArrayList<Integer> list = new ArrayList(Collections.singleton(nums1)) ;
//        list = (ArrayList<Integer>) list.subList(nums1.length-m-n-1,nums1.length);
//        nums1 = list.toArray();
    }
    //剑指 Offer 68 - II. 二叉树的最近公共祖先  有BUG
    class Info{
        boolean find01;
        boolean find02;
        TreeNode findAns;
        public Info(boolean f1,boolean f2,TreeNode node){
            this.find01 = f1;
            this.find02 = f2;
            this.findAns = node;
        }
    }
    public Info  process(TreeNode root,TreeNode f1, TreeNode f2){
        if (root == null)return new Info(false,false,null);
        Info left_info = process(root.left,f1,f2);
        Info right_info = process(root.left,f1,f2);
        if (left_info.findAns != null  )return new Info(true,true,left_info.findAns);
        if (right_info.findAns != null  )return new Info(true,true,right_info.findAns);
        if (left_info.find01 && right_info.find02)return new Info(true,true,root);
        if (left_info.find02 && right_info.find01)return new Info(true,true,root);
        return new Info(false,false,null);
    }
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
                 Info info = process(root,p,q);
                 return info.findAns;
    }
    //669. 修剪二叉搜索树
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null)return null;
        if (root.val < low)
            return trimBST(root.right,low,high);
        if (root.val > high)
            return trimBST(root.left,low,high);
        root.left =  trimBST(root.left,low,high);
        root.right = trimBST(root.right,low,high);
        return root;
    }
    //剑指 Offer 55 - I. 二叉树的深度
    public int maxDepth_1(TreeNode root) {
        if (root == null)return 0;
        int left = maxDepth_1(root.left) + 1;
        int right = maxDepth_1(root.right) + 1;
        return (left > right)? left + 1 : right + 1;
    }
//    559. N叉树的最大深度
    public int maxDepth(Node root) {
        if (root == null)return 0;
        int max = 1;
        for (Node child : root.children) {
            int temp = maxDepth(child) + 1;
            if(max < temp )
                max = temp;
        }
        return max;
    }
    //590. N叉树的后序遍历
    private List<Integer> post_list = new ArrayList<>();
    public List<Integer> postorder(Node root) {
        if (root == null)return post_list;
        for (Node child : root.children) {
            postorder(child);
        }
        post_list.add(root.val);
        return post_list;
    }
    //589. N叉树的前序遍历
    private List<Integer> pre_list = new ArrayList<>();
    public List<Integer> preorder(Node root) {
        //非递归
        Stack<Node> stack = new Stack<>();
        ArrayList<Integer> rs = new ArrayList<>();
        if (root == null)return rs;
        stack.push(root);
        while (!stack.isEmpty()){
            Node node = stack.pop();
            rs.add(node.val);
            for (int i = node.children.size()-1; i >= 0 ; i--) {
                stack.push(node.children.get(i));
            }
        }
        return rs;
//        if (root != null ){
//            pre_list.add(root.val);
//            for (Node child : root.children) {
//                preorder(child);
//            }
//        }
//        return pre_list;
    }

    //107. 二叉树的层次遍历 II
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> rs = new ArrayList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            ArrayList<Integer> temp = new ArrayList<>();
            int num = queue.size();
            TreeNode node = null;
            for (int i = 0; i < num; i++) {
                node = queue.poll();
                temp.add(node.val);
                if (node.left != null){
                    queue.add(node.left);
                }
                if (node.right != null){
                    queue.add(node.right);
                }
            }
            rs.add(0,temp);

        }
        return rs;
    }
    //108. 将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return findMid(nums,0,nums.length-1);
    }
    public TreeNode findMid(int[] nums, int left, int right){
        if (left > right)return null;
        int mid = (left + right) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = findMid(nums,left,mid-1);
        node.right = findMid(nums,mid+1,right);
        return node;
    }
    //
    public boolean isUnivalTree(TreeNode root) {
        if(root == null)return false;
        return  preNode(root,root.val);



    }
    public boolean preNode(TreeNode root,int val){
        if(root != null){
            if(root.val != val)return false;
            preNode(root.left,val);
            preNode(root.right,val);
        }
        return true;
    }
    //637. 二叉树的层平均值
    public List<Double> averageOfLevels(TreeNode root) {
        ArrayList<Double> rs = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        int clevel = 1;
        queue.add(root);
        while (!queue.isEmpty()) {
            Double sum = 0D;
            int num = queue.size();
            TreeNode node = null;
            for (int i = 0; i < num; i++) {
                node = queue.poll();
                sum += node.val;
                if (node.left != null) {
                    queue.add(node.left);
                }
                if (node.right != null) {
                    queue.add(node.right);
                }
            }
            rs.add(sum / num);
        }
        return rs;
    }
    //897. 递增顺序查找树
    Queue<TreeNode> query = new LinkedBlockingQueue<>();
    public TreeNode increasingBST(TreeNode root) {
        //让每个节点按照中序遍历顺序入队，递归构建右子树
        if (root == null) return null;
        midTree(root.left);
        query.add(root);
        midTree(root.right);
        TreeNode root_1 = query.poll();
        TreeNode temp = root_1;
        while (!query.isEmpty()){
             temp.right = query.poll();
             temp.left = null;
             temp = temp.right;
        }
        return root_1;
    }
    //中序遍历
    public void midTree(TreeNode root){
        if (root == null)return;
        midTree(root.left);
        query.add(root);
        midTree(root.right);
    }
    //    剑指 Offer 54. 二叉搜索树的第k大节点
    private ArrayList<Integer> list = new ArrayList<>();
    public int kthLargest(TreeNode root, int k) {
        mid(root);
        return list.get(list.size()-1-k);
    }
    public void mid(TreeNode root){
        if (root != null){
            mid(root.left);
            list.add(Integer.valueOf(root.val));
            mid(root.right);
        }
    }
    // Definition for a Node.
    class Node {
        public int val;
        public List<Node> children;
        public Node next;
        public Node random;

        public Node() {}

        public Node(int _val) {
            this.val = _val;
            this.next = null;
            this.random = null;
        }

        public Node(int _val, List<Node> _children) {
            val = _val;
            children = _children;
        }
    };
    //700. 二叉搜索树中的搜索
    public TreeNode searchBST(TreeNode root, int val) {
        if(root == null)
            return null;
        if (root.val == val)
            return root;
        else if (root.val < val)
            return searchBST(root.right,val);
        else return searchBST(root.left,val);
    }
    //    104. 二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null)return 0;
        int left = maxDepth(root.left) + 1;
        int right = maxDepth(root.right) + 1;
        return 1 + ((left > right)? left:right);
    }

    //226. 翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        if (root == null)return  null;
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }
    //938. 二叉搜索树的范围和
    public int rangeSumBST(TreeNode root, int L, int R) {
        if(root == null)return 0;
        return  rangeSumBST(root.left,L,R) + rangeSumBST(root.right,L,R)
                + ((root.val <= R && root.val >= L)? root.val : 0);
    }
    //617. 合并二叉树
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if(t1 == null)return t2;
        if (t2 == null)return t1;
        t1.val += t2.val;

        t1.left = mergeTrees(t1.left,t2.left);
        t1.right = mergeTrees(t1.right,t2.right);
        return t1;
    }
    //    1370. 上升下降字符串
    public String sortString(String s) {
        char[] chars = s.toCharArray();
        StringBuffer buffer = new StringBuffer();
        Arrays.sort(chars);
        //第一个放入的一定是最小的额
        buffer.append(chars[0]);
        char min = 'z',max = 'a';
        for (int j= 0; j < s.length(); j++) {
            int i ;
            for (i = 1 ; i < s.length()  && chars[i] > 'a' && chars[i] < 'z' ; i++) {
                if (min > chars[i]) min = chars[i];
            }//找到就把原位置的数置为’0‘
            chars[i] = '0';
            buffer.append(min);


        }


        return null;
    }
    //1491. 去掉最低工资和最高工资后的工资平均值
    public double average(int[] salary) {
        Arrays.sort(salary);
        Double sum = 0.0;
        for (int i = 1; i < salary.length-1; i++) {
            sum += salary[i];
        }
        return sum / (salary.length-2);
    }
    //    1528. 重新排列字符串
    public String restoreString(String s, int[] indices) {
        char[] rs = new char[s.length()];
        for (int i = 0; i < s.length(); i++) {
            rs[indices[i]] = s.charAt(i);
        }
        return new String(rs);
    }

    //242. 有效的字母异位词
    public boolean isAnagram(String s, String t) {
        char[] char_s = s.toCharArray();
        char[] char_t = t.toCharArray();
        Arrays.sort(char_s);
        Arrays.sort(char_t);
        Character k ;
        Integer v = null;
        if(s.length() != t.length())return false;
        HashMap<Character,Integer> map_s = new HashMap<>();
        HashMap<Character,Integer> map_t = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            if (map_s.containsKey(s.charAt(i))){
              v = map_s.get(s.charAt(i));
                map_s.put(s.charAt(i), ++v);
            }else map_s.put(s.charAt(i),1);
            if (map_t.containsKey(t.charAt(i))){
                v = map_t.get(t.charAt(i));
                map_t.put(t.charAt(i), ++v);
            }else map_t.put(t.charAt(i),1);
        }
        //比较集合中每个词的出现次数
       Integer v_2 = null;
        for (int i = 0; i < s.length(); i++) {
            k = s.charAt(i);
            v = map_s.get(k);
            v_2 = map_t.get(k);
            if (v_2 ==null || v == null || !v_2.equals(v))return false;
        }

        return true;
    }

    //234. 回文链表
    public boolean isPalindrome(ListNode head) {
            if( head == null)return true;
            ListNode mid = middleNode(head);
            //后半段链表头节点
            ListNode midHeader = mid.next;
            //断开前面的链表
            mid.next = null;

            ListNode header = null;
            ListNode temp = midHeader;
            ListNode buff = null;
            //反转后半段链表
            while (temp != null){
                buff = temp.next;
                temp.next = header;
                header = temp;
                temp = buff;
            }
            mid = header;
            header = head;
            while (mid != null &&  header != null){
                if (mid.val == header.val ){
                    mid = mid.next;
                    header = header.next;
                }else return false;
            }
            return true;
        }
    //876. 链表的中间结点
    public ListNode middleNode(ListNode head) {
        int length = 0;
        ListNode cur = head;
        while (cur != null){
            length++;
            cur = cur.next;
        }
        int mid = length / 2;
        cur = head;
        while (mid != 1){
            cur = cur.next;
            mid--;
        }
        return cur;
    }
//    203. 移除链表元素
    public ListNode removeElements(ListNode head, int val) {
        ListNode node = new ListNode(-1);
        node.next = head;
        ListNode curr = node;
        while (curr.next != null){
            if (curr.next.val == val){
                curr.next = curr.next.next;
            }
            else {
                curr  = curr.next;
            }
        }
        return node.next;
    }
    //160. 相交链表
    public ListNode getIntersectionNode1(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        int distance = 0;
        ListNode temp_a = headA,temp_b = headB;
        while (temp_a.next != null){
            distance++;
            temp_a = temp_a.next;
        }
        while (temp_b.next != null){
            distance--;
            temp_b = temp_b.next;
        }
        if (temp_a != temp_b)return null;
        //长链表指针给temp_a,短的给temp_b
        temp_a = distance > 0 ? headA : headB;
        temp_b = temp_a == headA ? headB : headA;
        distance = Math.abs(distance);
        while (distance > 0){
            temp_a = temp_a.next;
            distance--;
        }
        while (temp_a != temp_b){
            temp_a = temp_a.next;
            temp_b = temp_b.next;
        }
        return temp_a;
    }
    //83. 删除排序链表中的重复元素
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null)return head;
        ListNode temp = head;
        ListNode p = head;
        while (p != null){
            if (temp.val == p.val){
                p = p.next;
            }else {
                temp.next = p;
                temp = p;
            }
        }
        temp.next = null;
        return head;
    }

    //面试题 02.07. 链表相交
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int num = 0;
        ListNode p1 = headA,p2 = headB;
        while (p1 != null){
            num++;
            p1 = p1.next;
        }
        while (p2 != null){
            num--;
            p2 = p2.next;
        }
        p1 = headA;
        p2 = headB;
        //num > 0 p1 长 < 0 p2 长
        //长的先走num步长
        if (num > 0){
            while (num > 0){
                p1 = p1.next;
                num--;
            }
        }else {
            num = Math.abs(num);
            while (num > 0){
                p2 = p2.next;
                num--;
            }
        }
        while(p1 != null && p2 != null){
            if (p1 == p2) return p1;
            p1 = p1.next;
            p2 = p2.next;
        }
        return null;
    }
    //剑指 Offer 06. 从尾到头打印链表
    public int[] reversePrint(ListNode head) {
        ListNode node = head;
        ArrayList<Integer> list = new ArrayList<>();
        while (node != null) {
            list.add(node.val);
            node = node.next;
            }
        int[] arr = new int[list.size()];
        for (int i = list.size()-1,j = 0; i >= 0 ; i--,j++) {
            arr[j] = list.get(i);
        }
        return arr;
    }

    //141. 环形链表
    public boolean hasCycle(ListNode head) {
        if (head == null) return false;
        HashSet<ListNode> set = new HashSet<>();
        ListNode node = head;
        while (node != null){
            if (!set.contains(node)){
                set.add(node);
                node = node.next;
            }else return true;
        }
        return false;
    }
    //628. 三个数的最大乘积
    // 给定一个整型数组，在数组中找出由三个数组成的最大乘积，并输出这个乘积。
    public int maximumProduct(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        //用最小的两个负数之积与第二第三大的正数比较
        if ((nums[0] * nums[1]) < nums[n-2]*nums[n-3]){
            return  nums[n-2]*nums[n-3]*nums[n-1];
        }
        else {
            return nums[0] * nums[1] * nums[n - 1];
        }
    }
    //485. 最大连续1的个数
    public int findMaxConsecutiveOnes(int[] nums) {
//        public int findMaxConsecutiveOnes(int[] nums) {
//            int max = 0 , count = 0 ;
//            for(int i = 0 ; i < nums.length ; i++){
//                if(nums[i] == 1) count++;
//                else{
//                    if(count > max) max = count;
//                    count = 0;
//                }
//            }
//            if(count > max) max = count;
//            return max;
        int max = 0;
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 1)count++;
            else {
                if (max < count) {
                    max = count;
                    count = 0;
                }
            }
        }
        if (max < count)
            return count;
        return max;
    }
    //将两个升序链表合并成一个升序链表
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode newHead = new ListNode(0);
        ListNode tail = newHead;
        while (l1 != null && l2 != null){
            if (l1.val < l2.val){
                tail.next = l1;
                tail = l1;
                l1 = l1.next;
            }else {
                tail.next = l2;
                tail = l2;
                l2 = l2.next;
            }
        }
        tail.next = l1 == null ? l2:l1;
        return newHead.next;
    }
    // 移除重复的节点
    public ListNode removeDuplicateNodes(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        HashSet<Integer> setList = new HashSet<>();
        //构建一个头节点
        ListNode fakeHead = new ListNode(0 );
        fakeHead.next = head;

        //创建两个游标 一个指向头节点一个指向下一位
        ListNode pre =  fakeHead;
        ListNode node = pre.next;
        while(node != null){
            if (setList.contains(node.val)){
                node = node .next;
                pre.next = node;

            }else {
                setList.add(node.val);
                pre = node;
                node = node.next;
            }
        }
    return head;

    }

//
public ListNode reverseList(ListNode head) {
    //1 -> 2 -> 3 -> 4 -> null
    ListNode record = null;
    ListNode temp =  head.next;
    head.next = null;
    while (temp != null) {
        record = temp.next;
        temp.next = head;
        head = temp;
        temp = record;
    }
    return head;
}


//
public void deleteNode(ListNode node) {
 //a->b->c->d->e->f
    ListNode temp = null;
    while (node.next != null) {
        temp = node;
        node.val = temp.next.val;
        node = node.next;
    }
    temp.next = null;
}

// leetcode 
public int getDecimalValue(ListNode head) {
    List<Integer> list  = new ArrayList<>();
    while(head != null){
        list.add(head.val);
        head = head.next;
    }
    int rs = 0;
    for (int i = 0; i <= list.size() - 1; i++) {
        if (list.get(i) == 0) continue;
        rs += Math.pow(2, list.size() - i - 1) *list.get(i);
    }
    return rs;
}

public class ListNode {
         int val;
    ListNode next;
    ListNode() {}
     ListNode(int val) { this.val = val; }
     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    static class Student{
        String name;
        int score;
        public Student(String name,int score){
            this.name = name;
            this.score = score;
        }

        public String toString(){
            return "name  " + name + " score " + score;
        }
    }

  public static class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode(int x) { val = x; }
  }




    public static String changToChinese(char n){
        switch (n){
            case '0' :return "ling ";
            case '1' : return "yi ";
            case '2' : return "er ";
            case '3' : return "san ";
            case '4' : return "si ";
            case '5' : return "wu ";
            case '6' : return "liu ";
            case '7' : return "qi ";
            case '8' : return "ba ";
            case '9' : return "jiu ";
        }
        return  "";
    }



    public static int reverseNum(int num){
        int length = String.valueOf(num).length();
        int buffer = 0;
        for (int i = 0; i < length; i++) {
            buffer = num % 10;
            num = num / 10;
            return buffer;
        }
        return -1;
    }
    public static void erFen(int low,int high,int[] arr,int target  ){
        if (low > high)return;
        if(target == arr[0]){
            System.out.print(0);
            return;
        }else if (target == arr[arr.length-1]){
            System.out.print(arr.length-1);
            return;
        }
        int mid = (low + high) / 2;
        if(target == arr[mid]){
            System.out.print(mid);
            return;
        }
        else if (target < arr[mid])
            erFen(low,mid-1,arr,target);
        else
            erFen(mid+1,high,arr,target);

    }
    public static void bubble(int[] data, int length){
        boolean flag = false;
        for (int i = 0; i < length; i++) {
            for (int j = length - 1; j > i ; j--) {
                if (data[j] < data[j-1]){
                    int temp = data[j-1];
                    data[j-1] = data[j];
                    data[j] = temp;
                }
            }
            if (flag == false)return;
        }
    }
    public static boolean isSuShu(int  number){
        for (int i = 2; i <= Math.sqrt((double)number); i++) {
            if (number % i == 0)return false;
        }
        return true;
    }
    public static boolean isCircleNum(int number){
        int original = number;
        int reminder,now = 0;
        while (number != 0){
            reminder = number % 10;
            now = reminder + now * 10;
            number = number / 10;
        }
        if (now != original) return false;
        else {
            List<Integer> list = new ArrayList<>();
            int birn = 0;
            number = original;
            while(number != 0){
                birn = number % 2;
                number = number / 2;
                list.add(birn);

            }
            for (int i = 0,j = list.size()-1; i < j; i++,j--) {
                if (list.get(i) != list.get(j))return  false;
            }

        }
        return true;
    }

    public static void t1() {
        Scanner in = new Scanner(System.in);
        String str = in.next();
        String key = in.next();

        char key_int = key.toCharArray()[0];
        char[] buffer = str.toCharArray();
        int[] buffer_1 = new int[buffer.length];

        System.out.printf("加密后的");
        for (int i = 0; i < buffer.length; i++) {
            buffer_1[i] = (buffer[i] ^ key_int);
            System.out.print(buffer_1[i]);
        }
        System.out.println("解密后的");
        for (int j = 0; j < buffer_1.length; j++) {
            System.out.print((char) (buffer_1[j] ^ key_int));
        }
    }

    // 打印 * 平行四边形
    public static void solution123() {
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < i; k++) {
                System.out.print(" ");
            }
            for (int j = 0; j < 5; j++) {
                System.out.print("*" + " ");
            }
            System.out.println();
        }
    }

    // 打印金字塔
    public static void solution12() {
        System.out.println("please ---");
        Scanner in = new Scanner(System.in);
        int n = in.nextInt();
        // while(n != 0){
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < (40 - i); j++) {
                System.out.print("-");
            }
            for (int j = 0; j < (2 * i - 1); j++) {
                System.out.print("*");
            }
            System.out.println();
        }

        // }
        in.close();
    }

    // 水仙花数
    public static void solution2008_shuixian() {
        List<Integer> list = new ArrayList<>();
        for (int i = 100; i < 1000; i++) {
            // 个位数
            int a = i % 10;
            // 十位数
            int b = (i % 100) / 10;
            // 百位数
            int c = i / 100;
            if (i == a * a * a + b * b * b + c * c * c) {
                list.add(i);
            }

        }
        for (Integer l : list) {
            System.out.print("-" + l);
        }
    }

    // 求完全平方、二位数字相同
    public static void solution2007() {

        List<Integer> list = new ArrayList<>();
        for (int num = 100; num < 1000; num++) {
            int a = num % 10;// 个位数
            int c = num / 100;// 百位数
            int b = (num % 100) / 10;// 十位数
            if ((a == b || a == c || b == c) && fac(num)) {
                list.add(num);
            }
        }
        for (Integer l : list) {
            System.out.print("-" + l);
        }

    }

    /**
     * 判断一个数是不是完全平方
     */
    public static boolean fac(int num) {
        for (int i = 1; num > 0; i += 2) {
            num -= i;
        }
        if (num == 0)
            return true;
        else
            return false;
    }

    // 求数列第n项值
    public static int solution2008(int n) {
        if (n == 1 || n == 2)
            return 1;
        return solution2008(n - 1) + solution2008(n - 2);
    }

    /*
     * 将下标为奇数的小写字母转换成大写字母
     */
    public static void sloution2010() {
        Scanner in = new Scanner(System.in);
        String str = in.nextLine();
        in.close();
        char[] arr = str.toCharArray();
        for (int i = 0; i < arr.length; i++) {
            if (i % 2 == 1) {
                if (arr[i] <= 'z' && arr[i] >= 'a') {
                    arr[i] -= ('a' - 'A');
                }
            }
        }
        System.out.println(new String(arr));
    }

    public static int majorityElement1(int[] nums) {
        int mid = nums.length / 2;
        Map<Integer, Integer> myMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (myMap.containsKey(nums[i])) {
                myMap.put(nums[i], myMap.get(nums[i]) + 1);
            } else {
                myMap.put(nums[i], 1);
            }
        }
        Map.Entry<Integer, Integer> buff = null;
        for (Map.Entry<Integer, Integer> entry : myMap.entrySet()) {
            if (buff == null || entry.getValue() > buff.getValue()) {
                buff = entry;
         }
        }
        return buff.getKey();
    }

    private int keyWithMaxValue(Map<Integer, Integer> map) {
        Map.Entry<Integer, Integer> mEntry = null;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) { // Set<Map.Entry<K,V>>中有getKey和getValue方法
            if (mEntry == null || entry.getValue() > mEntry.getValue()) {
                mEntry = entry;
            }
        }
        return mEntry.getKey();
    }


}
