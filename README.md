# Multiple-Linear-Regression-by-using-Gradient-Descent
- 1 緒論

  - 1.1線性回歸的定義
  - 1.2單變量線性回歸
  - 1.3多變量線性回歸

- 2梯度下降
  - 2.1 cost function
  - 2.2 梯度下降：解決線性回歸的方法之一
  - 2.3 feature scaling：加快梯度下降執行速度的方法
  - 2.4 梯度下降的實現

- 3 梯度下降的擴展和比較

## 摘要:
本文首先說明線性回歸的定義，引出梯度下降法的需求和好處。然後分析了梯度下降的基本原理，給出基於matlab的具體實現方法。總結梯度下降算法的特點，並分析相關參數對算法運行結果的影響。



## 1 緒論
### 1.1線性回歸的定義
線性回歸，是利用數理統計中回歸分析，來確定兩種或兩種以上變量間相互依賴的定量關係的一種統計分析方法，運用十分廣泛。其表達形式為y = w'x+e，e為誤差服從均值為0的正態分佈。
具體來說，如果輸入x是列向量，目標y 是連續值（實數或連續整數），預測函數f(x)的輸出也是連續值。這種機器學習問題是回歸問題。如果我們定義f(x)是線性函數，f(x) = wTx + b, 這就是線性回歸問題（Linear Regression）。

### 1.2單變量線性回歸
單變量線性回歸就是從一個輸入值預測一個輸出值。輸入/輸出的對應關係就是一個線性函數。

### 1.3多變量線性回歸
多變量線性回歸就是說輸出值受多個輸入變量x影響，每個變量的影響大小用w(weight)來刻畫，w與x呈線性關係，然後把多個wx線性組合(w1x1,w2x2,,,)相加，再和輸出y的對應關係就是一個多變量線性回歸問題。注意：多變量線性回歸不是指y與X是線性的，而是指各自的w與x呈線性關係。所以最終的結果，在幾何上甚至可能呈現出曲線，而不是必須是直線。

## 2 梯度下降
### 2.1 cost function
線性回歸屬於監督學習，因此方法和監督學習是一樣的，先給定一個訓練集，根據這個訓練集學習出一個線性函數，然後測試這個函數是否足夠擬合訓練集數據，挑選出最好的函數（cost function最小）即可。

1)下面給出單變量線性回歸的模型：
怎麼樣能夠看出線性函數擬合的好不好呢？
我們需要使用到Cost Function（代價函數），代價函數越小，說明線性回歸地越好，和訓練集擬合地越好，最小就是0，即完全擬合。

![image](https://user-images.githubusercontent.com/97221948/149253700-3a19abdb-c972-4ba7-8679-fb1c55790ae0.png)

多變量的假設h表示為：

![image](https://user-images.githubusercontent.com/97221948/149254579-8e3515fc-52ff-4127-b91a-73870b4fc6eb.png)

代價函數：

![image](https://user-images.githubusercontent.com/97221948/149254602-7a98e36c-e702-40ff-860a-46ae4eb42371.png)

找出使得代價函數最小的一系列參數。
多變量線性回歸的批量梯度下降算法為：

![image](https://user-images.githubusercontent.com/97221948/149254754-a5084d02-2375-40e4-a108-0a44ca581a17.png)

求導後得到：

![image](https://user-images.githubusercontent.com/97221948/149254782-d208a356-e2c9-4b1c-a031-694a2c4e2a04.png)



2)舉個實際的例子：
我們想要根據房子的大小，預測房子的價格，給定如下數據集：

根據以上的數據集畫在圖上，如下圖所示：

我們需要根據這些點擬合出一條直線，使得cost Function最小。雖然我們現在還不知道Cost Function內部到底是什麼樣的，但是我們的目標是：給定輸入向量x，輸出向量y，theta向量，輸出Cost值。

3)Cost Function的用途：對假設的函數進行評價，cost function越小的函數，說明擬合訓練數據擬合的越好。
下圖詳細說明了當cost function為黑盒的時候，cost function 的作用。


4)但是我們肯定想知道cost Function的內部構造是什麼？因此我們下面給出公式：


其中：
表示向量x中的第i個元素；表示向量y中的第i個元素；
表示已知的假設函數； m 為訓練集的數量。

比如給定數據集(1,1)、(2,2)、(3,3)
則x = [1;2;3]，y = [1;2;3]（此處的語法為Octave語言的語法，表示3*1的矩陣）
如果我們預測theta0 = 0，theta1 = 1，則h(x) = x，則cost function：
J(0,1) = 1/(2*3) * [(h(1)-1)^2+(h(2)-2)^2+(h(3)-3)^2] = 0；
如果我們預測theta0 = 0，theta1 = 0.5，則h(x) = 0.5x，則cost function：
J(0,0.5) = 1/(2*3) * [(h(1)-1)^2+(h(2)-2)^2+(h(3)-3)^2] = 0.58；

如果theta0 一直為0， 則theta1與J的函數為：


如果有theta0與theta1都不固定(即都為變量)，則theta0、theta1、J 的函數為：

當然我們也能夠用二維的圖來表示，即等高線圖。


注意：如果是線性回歸，則costfunctionJ與theta0、theta1的函數一定是碗狀的，即只有一個最小點。

### 2.2 梯度下降：解決線性回歸的方法之一
但是又一個問題引出了，雖然給定一個函數，我們能夠根據cost function知道這個函數擬合的好不好，但是畢竟函數有這麼多，總不可能一個一個試吧？因此我們引出了梯度下降：能夠找出cost function函數的最小值；梯度下降原理：將函數比作一座山，我們站在某個山坡上，往四周看，從哪個方向向下走一小步，能夠下降的最快。當然解決問題的方法有很多，梯度下降只是其中一個，還有一種方法叫Normal Equation。

方法：
(1)先確定向下一步的步伐大小，我們稱為Learning rate；

(2)任意給定一個初始值 ![image](https://user-images.githubusercontent.com/97221948/149256849-7bfce4ae-2379-4282-8e1a-927c0976428e.png) ![image](https://user-images.githubusercontent.com/97221948/149256907-37fdb9a3-8d61-4c3f-b56a-78edef3accfd.png)；

(3)確定一個向下的方向，並向下走預先規定的步伐，並更新 ![image](https://user-images.githubusercontent.com/97221948/149256849-7bfce4ae-2379-4282-8e1a-927c0976428e.png) ![image](https://user-images.githubusercontent.com/97221948/149256907-37fdb9a3-8d61-4c3f-b56a-78edef3accfd.png)；

(4)當下降的高度小於某個定義的值，則停止下降。

算法：

特點：

(1)初始點不同，獲得的最小值也不同，因此梯度下降求得的只是局部最小值；

(2)越接近最小值時，下降速度越慢。

問題a：如果初始值就在local minimum的位置，則會如何變化？
答：因為已經在local minimum位置，所以derivative肯定是0，因此不會變化。
如果取到一個正確的值，則cost function應該越來越小。
問題b：怎麼取值？
答：隨時觀察值，如果cost function變小了，則ok，反之，則再取一個更小的值。

下圖就詳細的說明了梯度下降的過程：

![image](https://user-images.githubusercontent.com/97221948/149253895-c0ca2940-cb5b-442e-9779-869e272602fa.png)

從上面的圖可以看出：初始點不同，獲得的最小值也不同，因此梯度下降求得的只是局部最小值。

注意：下降的步伐大小非常重要，因為如果太小，則找到函數最小值的速度就很慢，如果太大，則可能會出現overshoot the minimum的現象；下圖就是overshoot minimum現象：

如果Learning rate取值後發現J function 增長了，則需要減小Learning rate的值。
因此我們能夠對cost function運用梯度下降，即將梯度下降和線性回歸進行整合，如下圖所示：

梯度下降是通過不停的迭代，而我們比較關注迭代的次數，因為這關係到梯度下降的執行速度，為了減少迭代次數，因此引入了Feature Scaling。

![image](https://user-images.githubusercontent.com/97221948/149253828-32a1449b-4c62-436c-bbbe-01c6c3b0b743.png)


```

clear
clc
gain10_f2000_1st_Data=importdata('C://Users//Effy//Desktop//Delta//211207_vibration//txt//gain10_f2000_1st.txt');
[gain10_f2000_1st_X] = inputdata_and_dataprocessing(gain10_f2000_1st_Data.data);
x1 = gain10_f2000_1st_X(:, 1);  % the size of the house
x2 = gain10_f2000_1st_X(:, 2);  % the number of bedrooms
x3 = gain10_f2000_1st_X(:, 3);  % the size of the house
x4 = gain10_f2000_1st_X(:, 4);  % the number of bedrooms
yT = gain10_f2000_1st_X(:, 5);  % the price of the house
y = yT.';
m = length(y); % number of training examples
% figure(1) %图1
% plot3(x1,x2,y, 'rx', 'MarkerSize', 10); % Plot the data
% xlabel('the size of the house'); % Set the x axis label
% ylabel('the number of bedrooms'); % Set the y axis label
% zlabel('the price of the house'); % Set the z axis label
% grid on;

%使x1、x2在[0,1]范围内
%x = [ones(1,m); 0.0001*x1.'; 0.1*x2.']; % 加第一列为全1，之后为x1、x2
x = [ones(1,m); x1.'; 0.0001*x2.'; 0.0001*x3.'; 0.0001*x4.']; % 加第一列为全1，之后为x1、x2
theta = zeros(5, 1); % initialize fitting parameters

iterations = 200; %迭代最大次数
alpha = 0.01; %学习率  %改变学习率，结果不一样
s = zeros(iterations, 1);  %代价函数中的累加值
J = zeros(iterations, 1);  %代价函数值

for k = 1:1:iterations 
    p = zeros(5, 1);  %迭代一次，累计清零
    for i = 1:1:m
        s(k) = s(k)+(theta.'*x(:,i)-y(i)).^2; %求J函数的累加
        %求偏导
        p = p+(theta.'*x(:,i)-y(:,i))*x(:,i); %对theta求偏导的累加   
    end         
    J(k) = s(k)/(2*m);  %代价函数
    theta = theta-(alpha/m)*p;  %更新theta参数
    if k>1  %为了下面k-1有索引
        if J(k-1)-J(k)<1e+2   %若误差小于10^2，则停止迭代         
             break;
        end
    end
end

theta  %输出显示theta的值

```


