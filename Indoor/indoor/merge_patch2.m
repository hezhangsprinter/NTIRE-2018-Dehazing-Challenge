clear all
% close all


list_all=dir('./our_cvprw_test3/*png');
image_index=1;

folder_name='./our_cvprw_test3sub/';
if not(exist(folder_name,'dir'))
        mkdir(folder_name)
end


for all_index=1:5
image1=zeros(2048,1024,3,5);

for i=1:5
    img1=imread(['./our_cvprw_test3/',num2str(image_index-1),'.png']);
    img1=imresize(img1,[1024,1024]);
    image_index=image_index+1;
    start=(i-1)*256+1;
    end1=(i-1)*256+1024;
%     image1(1:1024,start:end1,1:3,i)=img1;
    image1(start:end1,1:1024,1:3,i)=img1;    
end

zz2=sum(image1,4);

zz3=zeros(size(zz2));
start2=1;

ratio=[1,2,3,4,4,3,2,1];


zz3=zeros(2048,1024,3,8);

folder=cell(1,8);
for index=1:8
    start2=256*(index-1)+1;
    zz4=zz2(start2:start2+255,1:1024,1:3)/ratio(index);
    zz3(start2:start2+255,1:1024,1:3,index)=zz4;
end

zz4=sum(zz3,4);



size_all=[2833,4657;2833,4657;3052,4706;3122,4576;3122,4776];
% size_all=[2833,4657;2833,4657;2833,4657;2833,4657;2833,4657];
size_all=[2882,4476;2902,4086;3052,4706;3122,4576;3122,4776];

    img1=zz4/255;
    img1=imresize(img1,size_all(all_index,:));

    imwrite(img1, ['./our_cvprw_test3sub/',num2str(all_index+25),'.png']);


end
