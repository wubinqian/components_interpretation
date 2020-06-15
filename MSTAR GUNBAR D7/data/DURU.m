clear all;
close all;
clc;
%--------------------------------------------------trainAangle--------------------------%
% 
%     for j=1:72
%         a(j,1)=5*j-5
%     end
% %     c=[a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a];
%     c=[a;a;a;a;a;a;a;a;a;a;a;a;a;a;a;a];
% 
%------------------------------------------traindata---------------------------------%
% filepath='D:\tankewu\simulatedata\testdata\';
% 
% for k=1:16
%     temp=load([filepath,'data',num2str(k),'.mat']);
%     matname = char(fieldnames(temp));% 获取结构成员名称
%     data{k} = getfield(temp,matname);% 获取该名称下的成员内容，用  temp.usertrj_stay 也可以获得structure的值 data是一个cell
% end
% 
%     a1=cell2mat(data(1));
%     a2=cell2mat(data(2));
%     a3=cell2mat(data(3));
%     a4=cell2mat(data(4));
%     a5=cell2mat(data(5));
%     a6=cell2mat(data(6));
%     a7=cell2mat(data(7));
%     a8=cell2mat(data(8));
%     a9=cell2mat(data(9));
%     a10=cell2mat(data(10));
%     a11=cell2mat(data(11));
%     a12=cell2mat(data(12));
%     a13=cell2mat(data(13));
%     a14=cell2mat(data(14));
%     a15=cell2mat(data(15));
%     a16=cell2mat(data(16));
%     a17=cell2mat(data(17));
%     a18=cell2mat(data(18));
%     a19=cell2mat(data(19));
%     a20=cell2mat(data(20));
%     a21=cell2mat(data(21));
%     a22=cell2mat(data(22));
%     a23=cell2mat(data(23));
%     a24=cell2mat(data(24));
%     a25=cell2mat(data(25));
%     a26=cell2mat(data(26));
%     a27=cell2mat(data(27));
%     a28=cell2mat(data(28));
%     a29=cell2mat(data(29));
%     a30=cell2mat(data(30));
%     a31=cell2mat(data(31));
%     a32=cell2mat(data(32));
% 
%     c=[a1;a2;a3;a4;a5;a6;a7;a8;a9;a10;a11;a12;a13;a14;a15;a16;a17;a18;a19;a20;a21;a22;a23;a24;a25;a26;a27;a28;a29;a30;a31;a32];


%-----------------------------------------------------trainlable----------------------------------%
% for i=1:1152
%     trainLabel(i,1)=1;
% end
% 
% for i=1153:2304
%     trainLabel(i,1)=0;
% end

%--------------------------------------trainDangle-----------------------------------------%
% for i=1:1152
%     trainDangle(i,1)=15;
% end
% 
% trainDangle1=trainDangle(1:1152,1);
%-----------------------------------------------------------------------------------------%
load('trainLabel.mat');
% A=trainData(1000,:,:);
% B=reshape(A,64,64);
% figure;
% imshow(B);