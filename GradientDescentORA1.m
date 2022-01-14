gain10_f2000_1st_Data=importdata('C://Users//Effy//Desktop//Delta//211207_vibration//txt//gain10_f2000_1st.txt');
[gain10_f2000_1st_X] = inputdata_and_dataprocessing(gain10_f2000_1st_Data.data);

normal_con_training = gain10_f2000_1st_X(1 : round(length(gain10_f2000_1st_X)*0.8) ,:);
normal_con_testing = gain10_f2000_1st_X(round(length(gain10_f2000_1st_X)*0.8):length(gain10_f2000_1st_X),:);

x1 = normal_con_training(:, 1); 
x2 = normal_con_training(:, 2);  % sgn
x3 = normal_con_training(:, 3);  % speed
x4 = normal_con_training(:, 4);  % acc
yT = normal_con_training(:, 5);  %current
y = yT.';
m = length(y); % number of training examples
% minx2=min(x2);
% minx3=min(x3);
% minx4=min(x4);
% maxx2=max(x2);
% maxx3=max(x3);
% maxx4=max(x4);
stax2=std(x2);
stax3=std(x3);
stax4=std(x4);
meanx2=mean(x2);
meanx3=mean(x3);
meanx4=mean(x4);
norx2=(x2-meanx2)/(stax2);
norx3=(x3-meanx3)/(stax3);
norx4=(x4-meanx4)/(stax4);
%x = [ones(1,m); x1.'; 0.0001*x2.'; 0.0001*x3.'; 0.0001*x4.']; 
x = [ones(1,m);norx2.'; norx3'; norx4']; 
theta = zeros(4, 1); % initialize fitting parameters

iterations = 200; %迭代次數
alpha = 0.01; %learning rate
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
        if J(k-1)-J(k)==0   %若誤差小於10^2，則停止迭代         
             break;
        end
    end
end

