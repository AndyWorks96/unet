

class sortTest:
    def void sort(int[]arr, String[]path){

for (int i = 0; i < arr.length - 1; i++) {
int min = i;

// 每轮需要比较的次数 N-i
for (int j = i + 1; j < arr.length; j++) {
if (arr[j] < arr[min]) {
// 记录目前能找到的最小值元素的下标
min = j;
}
}

// 将找到的最小值和i位置所在的值进行交换
if (i != min) {
int tmp = arr[i];
String p=path[i];
arr[i] = arr[min];
arr[min] = tmp;
path[i]=path[min];
path[min]=p;
}

}
for (int i=0;i < arr.length;i++){
System.out.println(arr[i]);
}
for (int i=0;i < path.length;i++){
System.out.println(path[i]);
}
};

public static void main(String[] args) {

int[]a={3, 1, 10, 5, 5};
String[]b={"a/", "z", "mn", "xyx", "ha "};
sort(a, b);
}
}
