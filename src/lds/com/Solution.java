package lds.com;

import java.util.List;

public class Solution {
    public static void main(String[] args) {
        addBinary("1010","1011");
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
}
