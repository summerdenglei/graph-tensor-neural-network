%% generateing traning and testing data
clear
load Abilene_288_168_144_single
T=double(T);
data=zeros(168,12,24,144);
for j=1:168
    b=zeros(12,24,144);
    a=squeeze(T(:,j,:));
    for i=1:24
        a1=a((i-1)*12+1:(i-1)*12+12,:);
        b(:,i,:)=a1;
    end
    data(j,:,:,:)=b;
end
idx=randperm(168);
data1=data(idx,:,:,:);
sub_data=single(data1(1:140,:,:,:)); % training data
% sub_data=single(data1(141:168,:,:,:)); %test data
