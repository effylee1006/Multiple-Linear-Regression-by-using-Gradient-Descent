function [X] = inputdata_and_dataprocessing(Data)
verf =Data(:,2)*0.1;
origin_Y=Data(:,1)*0.1;
% plot(verf);
a=0;
change_point = 0;
Y=[];
spd=[];
low =[];
high =[];
new_verf2 =[];
new_Y =[];
new_Y2 =[];

for i = 1:(length(verf))
          
            if(verf(i)>-300 & verf(i)<-100)
                
                a=a+1;
                new_verf = verf(a:length(verf));
                new_Y = origin_Y(a:length(origin_Y));
                   for(j=1:length(new_verf))
                       if(new_verf(j)>=200)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                   end
                low = find(new_verf2 <-199.5 & new_verf2 >-200.5);
                for(i=1:length(new_verf2))
                    if(abs(new_verf2(1)-new_verf2(i))>=100)
                        change_point=i;
                        break;
                    end
                end

                wave = low(1)-change_point;
                
                if((4*change_point+3*wave)<length(new_Y2))
                    spd = new_verf2(1:(4*change_point+3*wave));
                    Y = new_Y2(1:(4*change_point+3*wave));
                else
                    spd = new_verf2;
                    Y = new_Y2;
                end
                                    
            elseif(verf(i)>-450 & verf(i)<-350)
                
                a=a+1;
                new_verf = verf(a:length(verf));
                new_Y = origin_Y(a:length(origin_Y));
                    for(j=1:length(new_verf))
                       if(new_verf(j)>=400)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                   end
                low = find(new_verf2 <-399.5 & new_verf2 >-400.5);
                for(i=1:length(new_verf2))
                    if(abs(new_verf2(1)-new_verf2(i))>=100)
                    change_point=i;
                    break;
                    end
                end

                wave = low(1)-change_point;
                if((4*change_point+3*wave)<length(new_Y2))
                    spd = new_verf2(1:(4*change_point+3*wave));
                    Y = new_Y2(1:(4*change_point+3*wave));
                else
                    spd = new_verf2;
                    Y = new_Y2;
                end
                       
            elseif(verf(i)>-650 & verf(i)<-550)
                
                a=a+1;
                new_verf = verf(a:length(verf));
                new_Y = origin_Y(a:length(origin_Y));
                   for(j=1:length(new_verf))
                       if(new_verf(j)>=600)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                   end
                 low = find(new_verf2 <-599.5 & new_verf2 >-600.5);
                 for(i=1:length(new_verf2))
                     if(abs(new_verf2(1)-new_verf2(i))>=100)
                        change_point=i;
                        break;
                     end
                 end

                wave = low(1)-change_point;
                if((4*change_point+3*wave)<length(new_Y2))
                    spd = new_verf2(1:(4*change_point+3*wave));
                    Y = new_Y2(1:(4*change_point+3*wave));
                else
                    spd = new_verf2;
                    Y = new_Y2;
                end
                                  
            elseif(verf(i)>-850 & verf(i)<-750)
                
                a=a+1;
                new_verf = verf(a:length(verf));
                new_Y = origin_Y(a:length(origin_Y));
                   for(j=1:length(new_verf))
                       if(new_verf(j)>= 800)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                   end
                 low = find(new_verf2 <-799.5 & new_verf2 >-800.5);
                 for(i=1:length(new_verf2))
                    if(abs(new_verf2(1)-new_verf2(i))>=100)
                        change_point=i;
                        break;
                    end
                 end

                wave = low(1)-change_point;
                if((4*change_point+3*wave)<length(new_Y2))
                    spd = new_verf2(1:(4*change_point+3*wave));
                    Y = new_Y2(1:(4*change_point+3*wave));
                else
                    spd = new_verf2;
                    Y = new_Y2;
                end
   
            elseif(verf(i)>-1050 & verf(i)<-950)
                
                a=a+1;
                new_verf = verf(a:length(verf));
                new_Y = origin_Y(a:length(origin_Y));
                   for(j=1:length(new_verf))
                       if(new_verf(j)>=1000)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                   end 
                 low = find(new_verf2 <-999.5 & new_verf2 >-1000.5);
                 for(i=1:length(new_verf2))
                    if(abs(new_verf2(1)-new_verf2(i))>=100)
                        change_point=i;
                        break;
                    end
                 end

                wave = low(1)-change_point;
               if((4*change_point+3*wave)<length(new_Y2))
                    spd = new_verf2(1:(4*change_point+3*wave));
                    Y = new_Y2(1:(4*change_point+3*wave));
                else
                    spd = new_verf2;
                    Y = new_Y2;
                end
                 
            elseif(verf(i)>150 & verf(i)<250)
                
                 a=a+1;
                 new_verf = verf(a:length(verf));
                 new_Y = origin_Y(a:length(origin_Y));
                    for(j=1:length(new_verf))
                       if(new_verf(j)<=-200)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                    end
                 high = find(new_verf2 >199.5 & new_verf2 <200.5);
                 for(i=1:length(new_verf2))
                    if(abs(new_verf2(1)-new_verf2(i))>=100)
                        change_point=i;
                        break;
                    end
                end

                wave = high(1)-change_point;
                if((5*change_point+4*wave)<length(new_Y2))
                    spd = new_verf2(change_point+wave:(5*change_point+4*wave));
                    Y = new_Y2(change_point+wave:(5*change_point+4*wave));
                else
                    spd = new_verf2(change_point+wave:length(new_verf2));
                    Y = new_Y2(change_point+wave:length(new_verf2));
                end
                                    
            elseif(verf(i)>350 & verf(i)<450)
                
                 a=a+1;
                 new_verf = verf(a:length(verf));
                 new_Y = origin_Y(a:length(origin_Y));
                    for(j=1:length(new_verf))
                       if(new_verf(j)<=-400)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                    end
                 high = find(new_verf2 >399.5 & new_verf2 <400.5);
                 for(i=1:length(new_verf2))
                    if(abs(new_verf2(1)-new_verf2(i))>=100)
                        change_point=i;
                        break;
                    end
                end

                wave = high(1)-change_point;
                if((5*change_point+4*wave)<length(new_Y2))
                    spd = new_verf2(change_point+wave:(5*change_point+4*wave));
                    Y = new_Y2(change_point+wave:(5*change_point+4*wave));
                else
                    spd = new_verf2(change_point+wave:length(new_verf2));
                    Y = new_Y2(change_point+wave:length(new_verf2));
                end
                
            elseif(verf(i)>550 & verf(i)<650)
                
                a=a+1;
                 new_verf = verf(a:length(verf));
                 new_Y = origin_Y(a:length(origin_Y));
                    for(j=1:length(new_verf))
                       if(new_verf(j)<=-600)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                    end
                 high = find(new_verf2 >599.5 & new_verf2 <600.5);
                 for(i=1:length(new_verf2))
                    if(abs(new_verf2(1)-new_verf2(i))>=100)
                        change_point=i;
                        break;
                    end
                end

                wave = high(1)-change_point;
                if((5*change_point+4*wave)<length(new_Y2))
                    spd = new_verf2(change_point+wave:(5*change_point+4*wave));
                    Y = new_Y2(change_point+wave:(5*change_point+4*wave));
                else
                    spd = new_verf2(change_point+wave:length(new_verf2));
                    Y = new_Y2(change_point+wave:length(new_verf2));
                end
                
            elseif(verf(i)>750 & verf(i)<850)
                   
                a=a+1;
                 new_verf = verf(a:length(verf));
                 new_Y = origin_Y(a:length(origin_Y));
                    for(j=1:length(new_verf))
                       if(new_verf(j)<=-800)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                    end
                 high = find(new_verf2 >799.5 & new_verf2 <800.5);
                 for(i=1:length(new_verf2))
                    if(abs(new_verf2(1)-new_verf2(i))>=100)
                        change_point=i;
                        break;
                    end
                 end
                wave = high(1)-change_point;
                if((5*change_point+4*wave)<length(new_Y2))
                    spd = new_verf2(change_point+wave:(5*change_point+4*wave));
                    Y = new_Y2(change_point+wave:(5*change_point+4*wave));
                else
                    spd = new_verf2(change_point+wave:length(new_verf2));
                    Y = new_Y2(change_point+wave:length(new_verf2));
                end
                
                elseif(verf(i)>950 & verf(i)<1050)
                   
                a=a+1;
                 new_verf = verf(a:length(verf));
                 new_Y = origin_Y(a:length(origin_Y));
                    for(j=1:length(new_verf))
                       if(new_verf(j)<=-1000)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                    end
                 high = find(new_verf2 >999.5 & new_verf2 <1000.5);
                 for(i=1:length(new_verf2))
                    if(abs(new_verf2(1)-new_verf2(i))>=100)
                        change_point=i;
                        break;
                    end
                 end
                wave = high(1)-change_point;
                if((5*change_point+4*wave)<length(new_Y2))
                    spd = new_verf2(change_point+wave:(5*change_point+4*wave));
                    Y = new_Y2(change_point+wave:(5*change_point+4*wave));
                else
                    spd = new_verf2(change_point+wave:length(new_verf2));
                    Y = new_Y2(change_point+wave:length(new_verf2));
                end
                
                elseif(max(verf)>600 & max(verf)<700 & verf(i)<45)
                
                a=a+1;
                 new_verf = verf(a:length(verf));
                 new_Y = origin_Y(a:length(origin_Y));
                    for(j=1:length(new_verf))
                       if(new_verf(j)<=-600)
                         new_verf2 = new_verf(j:length(new_verf));
                         new_Y2 =new_Y(j:length(new_Y));
                         break;
                       end
                    end
                 high = find(new_verf2 >599.5 & new_verf2 <600.5);
                 for(i=1:length(new_verf2))
                    if(abs(new_verf2(1)-new_verf2(i))>=100)
                        change_point=i;
                        break;
                    end
                 end
                wave = high(1)-change_point;
                if((5*change_point+4*wave)<length(new_Y2))
                    spd = new_verf2(change_point+wave:(5*change_point+4*wave));
                    Y = new_Y2(change_point+wave:(5*change_point+4*wave));
                else
                    spd = new_verf2(change_point+wave:length(new_verf2));
                    Y = new_Y2(change_point+wave:length(new_verf2));
                end
                
            else
                break;
            end                
end

fsInner = 8000;
temp2 =[];
upper_spd = max(spd);
lower_spd = min(spd);

for i=1:length(spd)
    if(spd(i) < upper_spd*0.01 & spd(i) >lower_spd*0.01)
    temp2(i) = 1;
    else
    temp2(i) = 0;
    end
end

index2 = find(temp2==1);
spd(index2,:) = [];
Y(index2,:) = [];
acc =[];
sgn =[];
average_acc =[];
average_acc_new =[];

for i=1:(length(spd)-1)
    acc(i) = (spd(i+1)-spd(i));
end

acc = [acc'];

for i=1:(length(spd))
    if (spd(i)>0)
        sgn(i) = 1;
    elseif(spd(i)<0)
        sgn(i) = -1;
    end
end

for i=1:(length(acc)-32)
    mean_seg = acc(i:i+31);
    temp =  mean(mean_seg);
    average_acc(i) = temp;
end

average_acc_new(1)= 0;
for i=1:length(average_acc)
    average_acc_new(i+1) = average_acc(i);
end
for i=length(average_acc)+1:length(Y)
    average_acc_new(i)=0;
end

average_acc_new = [average_acc_new'];
sgn=[sgn'];

X=[ones(size(Y)) sgn spd average_acc_new*960000 Y];
upper_average_acc_new = max(average_acc_new)*0.9;
lower_average_acc_new = min(average_acc_new)*0.9;

temp = [];
for i=1:length(average_acc_new)
    if(average_acc_new(i) < lower_average_acc_new)
    temp(i) = 1;
    elseif(average_acc_new(i) > upper_average_acc_new)
    temp(i) = 1;
    end
end
index = find(temp == 1);
X(index,:) = [];
temp3 = [];

for i=1:length(X)
    if(X(i,4) == 0)
    temp3(i) = 1;
    end
end

index3 = find(temp3 == 1);
X(index3,:) = [];

end