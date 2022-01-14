# Multiple-Linear-Regression-by-using-Gradient-Descent
- 1 緒論

  - 1.1線性回歸的定義
  - 1.2單變量線性回歸
  - 1.3多變量線性回歸

- 2 梯度下降
  - 2.1 cost function
  - 2.2 梯度下降執行步驟
  - 2.3 梯度下降法應用於估計多變量迴歸模型參數
  - 2.4 梯度下降的Matlab實現
  
- 3 梯度下降的擴展和比較
  - 3.1 最小平方法
  - 3.2 最小平方法與梯度下降法之比較

## 摘要:
本文主要探討如何利用梯度下降法優迴歸歸模型參數且比較使用最小平方法與梯度下降法估計最佳參數於迴歸模型。首先說明線性迴歸的定義，以及在統計機器學習中的意義和現實生活中的影響，引出梯度下降法的需求和好處。然後分析了梯度下降的基本原理，給出基於matlab的具體實現方法。總結梯度下降算法的特點，並分析相關參數對算法運行結果的影響。最後，擴展並比較了其他幾種算法(批梯度下降算法、增量梯度下降算法、最小二乘法)以及相關實現。



## 1 緒論
### 1.1線性迴歸的定義
線性迴歸，是利用數理統計中迴歸分析，來確定兩種或兩種以上變量間相互依賴的定量關係的一種統計分析方法，運用十分廣泛。其表達形式為y = w'x+e，e為誤差服從均值為0的正態分佈。
具體來說，如果輸入x是列向量，目標y是連續值（實數或連續整數），預測函數f(x)的輸出也是連續值。這種機器學習問題是迴歸問題。如果我們定義f(x)是線性函數，f(x) = wTx + b, 這就是線性回歸問題（Linear Regression）。

### 1.2單變量線性迴歸
單變量線性回歸就是從一個輸入值預測一個輸出值。輸入/輸出的對應關係就是一個線性函數。

### 1.3多變量線性迴歸
多變量線性迴歸為輸出值受多個輸入變量x影響，每個變量的影響大小用w(weight)來刻畫，w與x呈線性關係，然後把多個wx線性組合(w1x1,w2x2,,,)相加，再和輸出y的對應關係就是一個多變量線性迴歸問題。多變量線性迴歸不是指y與X是線性的，而是指各自的w與x呈線性關係。所以最終的結果，在幾何上甚至可能呈現出曲線，而不是必須是直線。

### 1.4 多變量線性迴歸應用於建構物理模型
![image](https://user-images.githubusercontent.com/97490448/149264726-a845dcbe-7a1a-4ae4-bf56-78086056d616.png)

Notation :
𝜏_𝑚: motor torque 
𝜏_𝑓: friction 
𝜏_𝑔: gravity 
𝐽: moment of inertia 
𝛼: angular acceleration 
𝜔: angular velocity 
𝑓_𝑐: Coulomb friction 
𝐵: viscosity coefficient 

因馬達出的力 = 庫倫摩擦力根據方向 + 黏滯力 + 慣性力 + 重力，即𝝉_𝒎=𝒔𝒈𝒏(𝝎)×𝒇_𝒄+𝝎×𝑩+𝑱×𝜶+𝝉_𝒈，而目前的訊號數據僅包含，速度及電流，故須透過預處理，𝒔𝒈𝒏(𝝎)及𝜶透過原始資料轉換出來，即可利用多變量線性迴歸。

## 2 梯度下降
### 2.1 cost function
線性迴歸屬於監督學習，先給定一個訓練集，根據這個訓練集學習出一個線性函數，測試這個函數訓練的好不好，挑選出最好的函數，即cost function最小。

1)下面給出單變量線性迴歸的模型：
我們需要使用到Cost Function（代價函數），代價函數越小，說明線性回歸地越好，和訓練集擬合程度越高，0為最小，即完全擬合。

![image](https://user-images.githubusercontent.com/97221948/149253700-3a19abdb-c972-4ba7-8679-fb1c55790ae0.png)

多變量的假設h表示為：

![image](https://user-images.githubusercontent.com/97221948/149254579-8e3515fc-52ff-4127-b91a-73870b4fc6eb.png)

代價函數：

![image](https://user-images.githubusercontent.com/97221948/149254602-7a98e36c-e702-40ff-860a-46ae4eb42371.png)

找出使得代價函數最小的一系列參數，而在利用梯度下降法來找解的過程，參數修正的方向是要往梯度的反方向走才能朝極小值得方向，參數才會往損失函數最小化的地方前進，進而找到最佳參數解


### 2.2 梯度下降執行步驟

雖然給定一個函數，我們能夠根據cost function了解參數估計的好壞，然而函數眾多，不易計算，故引入梯度下降法：找出cost function函數的最小值；梯度下降原理：將函數比作一座山，尋找往哪個方向，能夠下降的最快。

方法：

(1)先確定向下一步的步伐大小，即Learning rate；

(2)任意給定一個初始值；

(3)透過梯度下降計算方式，更新參數；

多變量線性回歸的批量梯度下降算法為：

![image](https://user-images.githubusercontent.com/97221948/149254754-a5084d02-2375-40e4-a108-0a44ca581a17.png)

  - 𝜶為學習率。
  - 學習率越大則下降幅度越多，越小則下降幅度越小。
  - 學習率過大無法收斂。
  - 學習率過小可能只找到局部最佳點。


求導後得到：

![image](https://user-images.githubusercontent.com/97221948/149254782-d208a356-e2c9-4b1c-a031-694a2c4e2a04.png)

(4)當下降的高度小於某個定義的值或達到使用者設定之迭代次數，則停止下降。

特點：

(1)初始點不同，獲得的最小值也不同，因此梯度下降求得的只是局部最小值；

(2)越接近最小值時，下降速度越慢。

下圖說明梯度下降的過程：

![image](https://user-images.githubusercontent.com/97221948/149253895-c0ca2940-cb5b-442e-9779-869e272602fa.png)

從上面的圖可以看出，初始點不同，獲得的最小值也不同，因此梯度下降求得的只是局部最小值。
若learning rate 設定太小，找到函數最小值的速度很慢，但若太大，可能會可能會出現overshoot the minimum的現象，下圖就是overshoot minimum現象：

梯度下降是通過不停的迭代，而我們比較關注迭代的次數，因為這關係到梯度下降的執行速度，為了減少迭代次數，因此引入了Feature Scaling。

![image](https://user-images.githubusercontent.com/97221948/149253828-32a1449b-4c62-436c-bbbe-01c6c3b0b743.png)

### 2.3梯度下降法應用於估計多變量迴歸模型參數

(1)定義目標 :

根據多變量物理模型之速度、摩擦力的方向、加速度，預測訊號電流，
我們需要根據這些點擬合出一條直線，使得cost Function最小，目標為：給定輸入向量x，輸出向量y，theta向量，輸出Cost值，
不斷更新參數，直到最小化損失函數為止。

(2)Cost Function的用途：
對假設的函數進行評價，cost function越小的函數，說明擬合訓練數據擬合的越好。

(3)Cost Function 結構介紹 :

x1 = feedrate2000 gain10 摩擦力方向 ;
x2 = feedrate2000 gain10 速度 ;
x3 = feedrate2000 gain10 加速度 ;
y = feedrate2000 gain10 電流 ;

以下為Cost Function 計算方式 :

![image](https://user-images.githubusercontent.com/97490448/149305286-4e944260-a6e9-48b1-96ca-3c2d42d557c5.png)

經過 m 次迭代，minimize cost function，輸出最佳參數。

### 2.4 梯度下降的實現

欲利用𝝉_𝒎, 𝒔𝒈𝒏(𝝎), 𝝎, 𝜶，來估計𝒇_𝒄, 𝑩, 𝑱,𝝉_𝒈四個參數，進而得到預測值電流，在這裡，我們使用梯度下降法，進行參數估計 :

```
[gain10_f2000_1st_X] = inputdata_and_dataprocessing(gain10_f2000_1st_Data.data);
x1 = G10_F2000_1st_X(:, 1); 
x2 = G10_F2000_1st_X(:, 2);  % sgn
x3 = G10_F2000_1st_X(:, 3);  % speed
x4 = G10_F2000_1st_X(:, 4);  % acc
yT = G10_F2000_1st_X(:, 5);  %current
y = yT.';
m = length(y); % number of training examples

%x = [ones(1,m); x1.'; 0.0001*x2.'; 0.0001*x3.'; 0.0001*x4.']; 
x = [ones(1,m); 0.0001*x2.'; 0.0001*x3.'; 0.0001*x4.']; 
theta = zeros(4, 1); % initialize fitting parameters

iterations = 100; %迭代次數
alpha = 0.05; %learning rate
s = zeros(iterations, 1);  %代價函數中的累加值
J = zeros(iterations, 1);  %代價函數值

for k = 1:1:iterations 
    p = zeros(4, 1);  %迭代一次，累計清零
    for i = 1:1:m
       s(k) = s(k)+(theta.'*x(:,i)-y(i)).^2; %求J函数的累加
       %求偏導
       p = p+(theta.'*x(:,i)-y(:,i))*x(:,i); %对theta求偏導的累加     
    end         
    J(k) = s(k)/(2*m);  %代價函数
    theta = theta-(alpha/m)*p;  %更新theta參數
    if k>1  %為了下面k-1有索引
        if J(k-1)-J(k)==0   %若沒有誤差，則停止迭代         
             break;
        end
    end
end

```

## 3  梯度下降的擴展和比較
### 3.1 最小平方法
利用相同的物理模型，做最小平方法的參數估計，minimize 殘差平方和，當殘差平方和越小，代表參數估計的越好

![image](https://user-images.githubusercontent.com/97490448/149339344-8e894986-e7b9-414c-8366-b36458dd497f.png)

執行步驟 

(1)將物理模型簡化為 𝑦=𝑊∗𝑋 

(2)計算MSE 

![image](https://user-images.githubusercontent.com/97490448/149340548-e50351e1-da9a-43ee-946f-8222a5fd8cbe.png)

(3)利用微分，求一階導數為0 

![image](https://user-images.githubusercontent.com/97490448/149340773-8551ea16-166e-49bd-b2a2-758f5e1fe1e7.png)

(4)移項後，得到β  

![image](https://user-images.githubusercontent.com/97490448/149340897-91687201-61ff-4434-b127-b18760da5f13.png)

### 3.2 梯度下降法與最小平方法之比較

一般估計參數經常使用到最小平方法，本節欲探討最小平方法及梯度下降法的相同及相異之處。

(1)相同處

-最小平方法跟梯度下降法都是通過求導數來求損失涵數的最小值  

-給定已知數據的前提下估計參數，再給訂新的數據進行估算 

-使估計值和實際值的殘差為最小

(2)相異處

-最小平方法求導數後，找出global minimum，非迭代方法 ;

-梯度下降法在使用者定義迭代次數後，找 local minimum，對初始點及learning rate的設定很敏感

## Reference

https://blog.csdn.net/shaguabufadai/article/details/72858293

